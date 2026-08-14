//! Directory-based implementation of storage traits.

use fs2::FileExt as Fs2FileExt;
use locking::*;
use std::collections::HashMap;
use std::io::{self, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;

use async_trait::async_trait;

use super::consts::{FILENAMES, FILENAME_ENUM_MAP};

pub use tdb_succinct::storage::file::*;

use super::*;

const PREFIX_DIR_SIZE: usize = 3;
const IMPORT_MARKER: &str = ".terminus-store-import-tx";
const IMPORT_COMMIT_DIR: &str = ".terminus-store-import-commits";
const IMPORT_LOCK: &str = ".terminus-store-import.lock";
const LAYER_LOCK_SUFFIX: &str = ".terminus-store-write.lock";

#[cfg(test)]
fn injected_crash(point: &str) {
    if std::env::var("TERMINUS_STORE_DIRECTORY_CRASH_POINT").as_deref() == Ok(point) {
        std::process::abort();
    }
}

#[derive(Clone)]
pub struct DirectoryLayerStore {
    path: PathBuf,
}

impl DirectoryLayerStore {
    pub fn new<P: Into<PathBuf>>(path: P) -> DirectoryLayerStore {
        DirectoryLayerStore { path: path.into() }
    }

    fn layer_path(&self, name: [u32; 5]) -> PathBuf {
        let mut path = self.path.clone();
        let name = name_to_string(name);
        path.push(&name[0..PREFIX_DIR_SIZE]);
        path.push(name);
        path
    }

    fn import_commit_path(&self, token: &str) -> io::Result<PathBuf> {
        string_to_name(token)?;
        Ok(self
            .path
            .join(IMPORT_COMMIT_DIR)
            .join(format!("{token}.commit")))
    }

    fn layer_lock_path(&self, name: [u32; 5]) -> PathBuf {
        let layer = self.layer_path(name);
        let file_name = layer.file_name().unwrap().to_string_lossy();
        layer.with_file_name(format!(".{file_name}{LAYER_LOCK_SUFFIX}"))
    }

    async fn sync_directory(path: &std::path::Path) -> io::Result<()> {
        if cfg!(unix) {
            fs::OpenOptions::new()
                .read(true)
                .open(path)
                .await?
                .sync_all()
                .await?;
        }
        Ok(())
    }

    async fn ensure_child_directory(
        &self,
        parent: &std::path::Path,
        child: &std::path::Path,
    ) -> io::Result<()> {
        match fs::create_dir(child).await {
            Ok(()) => Self::sync_directory(parent).await,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                if fs::metadata(child).await?.is_dir() {
                    Ok(())
                } else {
                    Err(error)
                }
            }
            Err(error) => Err(error),
        }
    }

    async fn ensure_prefix(&self, name: [u32; 5]) -> io::Result<PathBuf> {
        let layer = self.layer_path(name);
        let prefix = layer.parent().unwrap().to_path_buf();
        self.ensure_child_directory(&self.path, &prefix).await?;
        Ok(prefix)
    }

    async fn acquire_layer_lock(&self, name: [u32; 5]) -> io::Result<std::fs::File> {
        let prefix = self.ensure_prefix(name).await?;
        let path = self.layer_lock_path(name);
        let (file, created) = tokio::task::spawn_blocking(move || {
            let created = !path.exists();
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(path)?;
            Fs2FileExt::lock_exclusive(&file)?;
            Ok::<_, io::Error>((file, created))
        })
        .await
        .map_err(|error| io::Error::new(io::ErrorKind::Other, error))??;
        if created {
            Self::sync_directory(&prefix).await?;
        }
        Ok(file)
    }

    async fn quarantine_layer_locked(&self, layer: &std::path::Path) -> io::Result<()> {
        let parent = layer.parent().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "layer path has no parent")
        })?;
        let name = layer.file_name().unwrap().to_string_lossy();
        let trash = parent.join(format!(".{name}.recovery-{:016x}", rand::random::<u64>()));
        match rename_directory_noreplace(layer.to_path_buf(), trash.clone()).await {
            Ok(()) => {
                // The visible name is gone before the marker-containing tree is
                // destroyed. Persist that fact before deleting the quarantine.
                Self::sync_directory(parent).await?;
                #[cfg(test)]
                injected_crash("during_quarantine");
                fs::remove_dir_all(&trash).await?;
                Self::sync_directory(parent).await
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn acquire_import_lock(&self) -> io::Result<std::fs::File> {
        fs::create_dir_all(&self.path).await?;
        let path = self.path.join(IMPORT_LOCK);
        let file = tokio::task::spawn_blocking(move || {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(path)?;
            Fs2FileExt::lock_exclusive(&file)?;
            Ok::<_, io::Error>(file)
        })
        .await
        .map_err(|error| io::Error::new(io::ErrorKind::Other, error))??;
        Self::sync_directory(&self.path).await?;
        Ok(file)
    }

    async fn import_visible(&self, layer_path: &std::path::Path) -> io::Result<bool> {
        let marker = layer_path.join(IMPORT_MARKER);
        let token = match fs::read_to_string(marker).await {
            Ok(token) => token,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(true),
            Err(error) => return Err(error),
        };
        let commit = self.import_commit_path(token.trim())?;
        match fs::metadata(commit).await {
            Ok(metadata) => Ok(metadata.is_file()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    async fn recover_uncommitted_imports_locked(&self) -> io::Result<()> {
        let mut prefixes = match fs::read_dir(&self.path).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        while let Some(prefix) = prefixes.next_entry().await? {
            if !prefix.file_type().await?.is_dir() {
                continue;
            }
            let prefix_name = prefix.file_name();
            let Some(prefix_name) = prefix_name.to_str() else {
                continue;
            };
            if prefix_name.len() != PREFIX_DIR_SIZE
                || !prefix_name.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                continue;
            }
            let mut entries = fs::read_dir(prefix.path()).await?;
            while let Some(entry) = entries.next_entry().await? {
                let path = entry.path();
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with('.') && name.contains(".import-") {
                    fs::remove_dir_all(path).await?;
                    Self::sync_directory(&prefix.path()).await?;
                    continue;
                }
                if name.starts_with('.') && name.contains(".recovery-") {
                    fs::remove_dir_all(path).await?;
                    Self::sync_directory(&prefix.path()).await?;
                    continue;
                }
                if !entry.file_type().await?.is_dir() {
                    continue;
                }
                if !self.import_visible(&path).await? {
                    let id = string_to_name(&name)?;
                    let _layer_lock = self.acquire_layer_lock(id).await?;
                    if !self.import_visible(&path).await? {
                        self.quarantine_layer_locked(&path).await?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
async fn rename_directory_noreplace(source: PathBuf, target: PathBuf) -> io::Result<()> {
    tokio::task::spawn_blocking(move || {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let source = CString::new(source.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "nul in import path"))?;
        let target = CString::new(target.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "nul in import path"))?;
        let result = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                target.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    })
    .await
    .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?
}

#[cfg(not(target_os = "linux"))]
async fn rename_directory_noreplace(_source: PathBuf, _target: PathBuf) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace publication is unavailable on this platform",
    ))
}

#[async_trait]
impl PersistentLayerStore for DirectoryLayerStore {
    type File = FileBackedStore;
    async fn directories(&self) -> io::Result<Vec<[u32; 5]>> {
        let _lock = self.acquire_import_lock().await?;
        self.recover_uncommitted_imports_locked().await?;
        let mut prefixes = fs::read_dir(&self.path).await?;
        let mut result = Vec::new();
        while let Some(prefix) = prefixes.next_entry().await? {
            if !prefix.file_type().await?.is_dir() {
                continue;
            }
            let mut entries = match fs::read_dir(prefix.path()).await {
                Ok(entries) => entries,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            while let Some(entry) = entries.next_entry().await? {
                if !entry.file_type().await?.is_dir() || !self.import_visible(&entry.path()).await?
                {
                    continue;
                }
                let os_name = entry.file_name();
                let Some(name) = os_name.to_str() else {
                    continue;
                };
                if let Ok(id) = string_to_name(name) {
                    result.push(id);
                }
            }
        }

        Ok(result)
    }

    async fn create_named_directory(&self, name: [u32; 5]) -> io::Result<[u32; 5]> {
        let _lock = self.acquire_import_lock().await?;
        self.recover_uncommitted_imports_locked().await?;
        let path = self.layer_path(name);
        let prefix = self.ensure_prefix(name).await?;
        fs::create_dir(path).await?;
        Self::sync_directory(&prefix).await?;

        Ok(name)
    }

    async fn directory_exists(&self, name: [u32; 5]) -> io::Result<bool> {
        let mut p = self.path.clone();
        let name = name_to_string(name);
        p.push(&name[0..PREFIX_DIR_SIZE]);
        p.push(name);

        match fs::metadata(&p).await {
            Ok(m) if m.is_dir() => self.import_visible(&p).await,
            Ok(_) => Ok(false),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    async fn get_file(&self, directory: [u32; 5], name: &str) -> io::Result<Self::File> {
        if !self.directory_exists(directory).await? {
            return Err(io::Error::new(io::ErrorKind::NotFound, "layer not found"));
        }
        let mut p = self.path.clone();
        let dir_name = name_to_string(directory);
        p.push(&dir_name[0..PREFIX_DIR_SIZE]);
        p.push(dir_name);
        p.push(name);
        Ok(FileBackedStore::new(p))
    }

    async fn file_exists(&self, directory: [u32; 5], file: &str) -> io::Result<bool> {
        if !self.directory_exists(directory).await? {
            return Ok(false);
        }
        let mut p = self.path.clone();
        let dir_name = name_to_string(directory);
        p.push(&dir_name[0..PREFIX_DIR_SIZE]);
        p.push(dir_name);
        p.push(file);

        match fs::metadata(p).await {
            Ok(m) => Ok(m.is_file()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    async fn abort_directory(&self, directory: [u32; 5]) -> io::Result<()> {
        let mut path = self.path.clone();
        let name = name_to_string(directory);
        path.push(&name[0..PREFIX_DIR_SIZE]);
        path.push(name);
        let parent = path.parent().map(|value| value.to_path_buf());
        match fs::remove_dir_all(path).await {
            Ok(()) => match parent {
                Some(parent) => Self::sync_directory(&parent).await,
                None => Ok(()),
            },
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn publish_imported_layers_shared(
        &self,
        layers: Vec<([u32; 5], HashMap<String, bytes::Bytes>)>,
    ) -> io::Result<()> {
        let _lock = self.acquire_import_lock().await?;
        self.recover_uncommitted_imports_locked().await?;
        if layers.is_empty() {
            return Ok(());
        }
        let mut ids = std::collections::HashSet::with_capacity(layers.len());
        for (id, files) in &layers {
            if !ids.insert(*id) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "duplicate imported layer id",
                ));
            }
            if files
                .keys()
                .any(|name| !FILENAME_ENUM_MAP.contains_key(name.as_str()))
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown imported layer structure",
                ));
            }
            match fs::metadata(self.layer_path(*id)).await {
                Ok(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "refusing to publish over an existing layer",
                    ));
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
        let mut ordered_ids: Vec<_> = ids.iter().copied().collect();
        ordered_ids.sort();
        let mut layer_locks = Vec::with_capacity(ordered_ids.len());
        for id in ordered_ids {
            layer_locks.push(self.acquire_layer_lock(id).await?);
        }
        let token = name_to_string(rand::random());
        let commit_dir = self.path.join(IMPORT_COMMIT_DIR);
        self.ensure_child_directory(&self.path, &commit_dir).await?;
        let mut staged = Vec::with_capacity(layers.len());
        for (id, files) in layers {
            let target = self.layer_path(id);
            let parent = self.ensure_prefix(id).await?;
            let temporary = parent.join(format!(
                ".{}.import-{:016x}",
                name_to_string(id),
                rand::random::<u64>()
            ));
            let build = async {
                fs::create_dir(&temporary).await?;
                let mut marker = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(temporary.join(IMPORT_MARKER))
                    .await?;
                marker.write_all(token.as_bytes()).await?;
                marker.flush().await?;
                marker.sync_all().await?;
                for (name, bytes) in files {
                    let path = temporary.join(name);
                    let mut file = fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(path)
                        .await?;
                    file.write_all(&bytes).await?;
                    file.flush().await?;
                    file.sync_all().await?;
                }
                if cfg!(unix) {
                    fs::OpenOptions::new()
                        .read(true)
                        .open(&temporary)
                        .await?
                        .sync_all()
                        .await?;
                }
                Ok::<_, io::Error>(())
            }
            .await;
            if let Err(error) = build {
                let _ = fs::remove_dir_all(&temporary).await;
                for (_, path, _) in &staged {
                    let _ = fs::remove_dir_all(path).await;
                }
                return Err(error);
            }
            staged.push((id, temporary, target));
        }

        #[cfg(test)]
        injected_crash("after_staging_fsync");

        let mut published: Vec<PathBuf> = Vec::new();
        for (id, temporary, target) in &staged {
            if let Err(error) = rename_directory_noreplace(temporary.clone(), target.clone()).await
            {
                for published_target in &published {
                    let _ = self.quarantine_layer_locked(published_target).await;
                }
                for (_, staged_path, _) in &staged {
                    let _ = fs::remove_dir_all(staged_path).await;
                }
                return Err(error);
            }
            published.push(target.to_path_buf());
            #[cfg(test)]
            injected_crash("after_target_rename");
            let _ = id;
        }
        let mut parents: Vec<_> = published
            .iter()
            .filter_map(|path| path.parent().map(|parent| parent.to_path_buf()))
            .collect();
        parents.sort();
        parents.dedup();
        for parent in parents {
            Self::sync_directory(&parent).await?;
        }
        let commit = self.import_commit_path(&token)?;
        let temporary_commit = commit_dir.join(format!(".{token}.tmp"));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_commit)
            .await?;
        file.write_all(b"committed\n").await?;
        file.flush().await?;
        file.sync_all().await?;
        rename_directory_noreplace(temporary_commit, commit).await?;
        #[cfg(test)]
        injected_crash("after_commit_marker_rename");
        // The commit-marker rename is the logical commit point, but success is
        // reported only after its parent directory is durable. A failure after
        // this point is explicitly typed as indeterminate so callers do not
        // mistake it for a retry-safe pre-commit failure.
        #[cfg(test)]
        let post_commit_durability =
            if std::env::var_os("TERMINUS_STORE_DIRECTORY_FAIL_POST_COMMIT_FSYNC").is_some() {
                Err(io::Error::new(
                    io::ErrorKind::Other,
                    "injected post-commit directory fsync failure",
                ))
            } else {
                Self::sync_directory(&commit_dir).await
            };
        #[cfg(not(test))]
        let post_commit_durability = Self::sync_directory(&commit_dir).await;
        std::mem::drop(layer_locks);
        post_commit_durability
            .map_err(|error| indeterminate_import_commit_error("directory", error))
    }

    async fn write_proof_state_bytes(&self, directory: [u32; 5], bytes: &[u8]) -> io::Result<()> {
        let _lock = self.acquire_import_lock().await?;
        self.recover_uncommitted_imports_locked().await?;
        let mut directory_path = self.path.clone();
        let name = name_to_string(directory);
        directory_path.push(&name[0..PREFIX_DIR_SIZE]);
        directory_path.push(name);
        if !self.directory_exists(directory).await? {
            return Err(io::Error::new(io::ErrorKind::NotFound, "layer not found"));
        }
        let target = directory_path.join(FILENAMES.proof_state);
        let temporary = directory_path.join(format!(
            ".{}.tmp-{:016x}",
            FILENAMES.proof_state,
            rand::random::<u64>()
        ));
        let result = async {
            let mut options = fs::OpenOptions::new();
            options.write(true).create_new(true);
            let mut file = options.open(&temporary).await?;
            file.write_all(bytes).await?;
            file.flush().await?;
            file.sync_all().await?;
            rename_directory_noreplace(temporary.clone(), target).await?;
            if cfg!(unix) {
                let directory_file = fs::OpenOptions::new()
                    .read(true)
                    .open(&directory_path)
                    .await?;
                directory_file.sync_all().await?;
            }
            Ok::<_, io::Error>(())
        }
        .await;
        if result.is_err() {
            let _ = fs::remove_file(&temporary).await;
        }
        result
    }

    async fn replace_proof_state_bytes_if_matches(
        &self,
        directory: [u32; 5],
        expected: &[u8],
        replacement: &[u8],
    ) -> io::Result<()> {
        let _lock = self.acquire_import_lock().await?;
        self.recover_uncommitted_imports_locked().await?;
        let name = name_to_string(directory);
        let directory_path = self.path.join(&name[..PREFIX_DIR_SIZE]).join(name);
        let target = directory_path.join(FILENAMES.proof_state);
        if fs::read(&target).await?.as_slice() != expected {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "proof state changed during migration",
            ));
        }
        let temporary = directory_path.join(format!(
            ".{}.migrate-{:016x}",
            FILENAMES.proof_state,
            rand::random::<u64>()
        ));
        let result = async {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .await?;
            file.write_all(replacement).await?;
            file.flush().await?;
            file.sync_all().await?;
            fs::rename(&temporary, &target).await?;
            let directory_file = fs::OpenOptions::new()
                .read(true)
                .open(&directory_path)
                .await?;
            directory_file.sync_all().await
        }
        .await;
        if result.is_err() {
            let _ = fs::remove_file(&temporary).await;
        }
        result
    }

    async fn finalize(&self, directory: [u32; 5]) -> io::Result<()> {
        if cfg!(unix) {
            // ensure the underlying directory record is properly synchronized
            let mut directory_path = self.path.clone();
            let dir_name = name_to_string(directory);
            directory_path.push(&dir_name[0..PREFIX_DIR_SIZE]);
            directory_path.push(dir_name);

            let mut options = tokio::fs::OpenOptions::new();
            options.create(false);
            options.read(true);
            options.write(false);
            let dir_fd = options.open(directory_path).await?;
            dir_fd.sync_all().await?;
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct DirectoryLabelStore {
    path: PathBuf,
}

impl DirectoryLabelStore {
    pub fn new<P: Into<PathBuf>>(path: P) -> DirectoryLabelStore {
        DirectoryLabelStore { path: path.into() }
    }
}

fn get_label_from_data(name: String, data: &[u8]) -> io::Result<Label> {
    let s = String::from_utf8_lossy(&data);
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expected label file to have two lines. contents were ({:?})",
                lines
            ),
        ));
    }

    let version_str = &lines[0];
    let layer_str = &lines[1];

    let version = u64::from_str_radix(version_str, 10);
    if version.is_err() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expected first line of label file to be a number but it was {}",
                version_str
            ),
        ));
    }

    if layer_str.is_empty() {
        Ok(Label {
            name,
            layer: None,
            version: version.unwrap(),
        })
    } else {
        let layer = layer::string_to_name(layer_str)?;
        Ok(Label {
            name,
            layer: Some(layer),
            version: version.unwrap(),
        })
    }
}

async fn get_label_from_file<P: Into<PathBuf>>(path: P) -> io::Result<Label> {
    let path: PathBuf = path.into();
    let label = path.file_stem().unwrap().to_str().unwrap().to_owned();

    let mut file = LockedFile::open(path).await?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).await?;

    get_label_from_data(label, &data)
}

async fn get_label_from_exclusive_locked_file<P: Into<PathBuf>>(
    path: P,
) -> io::Result<(Label, ExclusiveLockedFile)> {
    let path: PathBuf = path.into();
    let label = path.file_stem().unwrap().to_str().unwrap().to_owned();

    let mut file = ExclusiveLockedFile::open(path).await?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).await?;

    let label = get_label_from_data(label, &data)?;
    file.seek(SeekFrom::Start(0)).await?;

    Ok((label, file))
}

#[async_trait]
impl LabelStore for DirectoryLabelStore {
    async fn labels(&self) -> io::Result<Vec<Label>> {
        let mut stream = fs::read_dir(self.path.clone()).await?;
        let mut result = Vec::new();
        while let Some(direntry) = stream.next_entry().await? {
            if direntry.file_type().await?.is_file() {
                let os_name = direntry.file_name();
                let name = os_name.to_str().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unexpected non-utf8 directory name",
                    )
                })?;
                if name.ends_with(".label") {
                    let label = get_label_from_file(direntry.path()).await?;
                    result.push(label);
                }
            }
        }

        Ok(result)
    }

    async fn create_label(&self, label: &str) -> io::Result<Label> {
        let mut p = self.path.clone();
        p.push(format!("{}.label", label));
        let contents = "0\n\n".to_string().into_bytes();
        match fs::metadata(&p).await {
            Ok(_) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "database already exists",
            )),
            Err(e) => match e.kind() {
                io::ErrorKind::NotFound => {
                    let mut file = ExclusiveLockedFile::create_and_open(p).await?;
                    file.write_all(&contents).await?;
                    file.flush().await?;
                    file.sync_all().await?;

                    Ok(Label::new_empty(label))
                }
                _ => Err(e),
            },
        }
    }

    async fn get_label(&self, label: &str) -> io::Result<Option<Label>> {
        let mut p = self.path.clone();
        p.push(format!("{}.label", label));

        match get_label_from_file(p).await {
            Ok(label) => Ok(Some(label)),
            Err(e) => match e.kind() {
                io::ErrorKind::NotFound => Ok(None),
                _ => Err(e),
            },
        }
    }

    async fn set_label_option(
        &self,
        label: &Label,
        layer: Option<[u32; 5]>,
    ) -> io::Result<Option<Label>> {
        let new_label = label.with_updated_layer(layer);
        let contents = match new_label.layer {
            None => format!("{}\n\n", new_label.version).into_bytes(),
            Some(layer) => {
                format!("{}\n{}\n", new_label.version, layer::name_to_string(layer)).into_bytes()
            }
        };

        let mut p = self.path.clone();
        p.push(format!("{}.label", label.name));
        let (retrieved_label, mut file) = get_label_from_exclusive_locked_file(p).await?;
        if retrieved_label == *label {
            // all good, let's a go
            file.truncate().await?;
            file.write_all(&contents).await?;
            file.flush().await?;
            file.sync_all().await?;
            Ok(Some(new_label))
        } else {
            Ok(None)
        }
    }

    async fn delete_label(&self, name: &str) -> io::Result<bool> {
        let mut p = self.path.clone();
        p.push(format!("{}.label", name));

        // We're not locking here to remove the file. The assumption
        // is that any concurrent operation that is done on the label
        // file will not matter. If it is a label read, a concurrent
        // operation will simply get the label contents, which
        // immediately afterwards become invalid. Similarly if it is
        // for a write, the write will appear to be succesful even
        // though the file will be gone afterwards. This is
        // indistinguishable from the case where the read/write and
        // the remove happened in reverse order.
        match tokio::fs::remove_file(p).await {
            Ok(()) => Ok(true),
            Err(e) => match e.kind() {
                io::ErrorKind::NotFound => Ok(false),
                _ => Err(e),
            },
        }
    }
}

/// A version of the directory label store that keeps all labels in
/// memory and doesn't lock.
///
/// This is useful for situations where we can be sure that only one
/// process is working with a set of label files. In that case, we can
/// keep all label files cached in memory in order to process reads as
/// quickly as possible, and we can perform writes without any sort of
/// file system locking.
pub struct CachedDirectoryLabelStore {
    path: PathBuf,
    labels: Arc<RwLock<HashMap<String, Label>>>,
}

impl CachedDirectoryLabelStore {
    /// Open a new label store.
    ///
    /// This will read in all label files on startup, which is why
    /// this is an async operation.
    pub async fn open<P: Into<PathBuf>>(path: P) -> io::Result<Self> {
        let path: PathBuf = path.into();
        let labels = get_all_labels_from_dir(&path).await?;

        Ok(Self {
            path,
            labels: Arc::new(RwLock::new(labels)),
        })
    }
}

async fn get_all_labels_from_dir(p: &PathBuf) -> io::Result<HashMap<String, Label>> {
    let mut result = HashMap::new();
    let mut entries = tokio::fs::read_dir(p).await?;

    while let Some(entry) = entries.next_entry().await? {
        if !entry.file_type().await?.is_file() {
            continue;
        }
        if let Ok(file_name) = entry.file_name().into_string() {
            if !file_name.ends_with(".label") {
                continue;
            }

            let label_name = file_name[..file_name.len() - 6].to_string();
            let label = get_label_from_file(entry.path()).await?;

            result.insert(label_name, label);
        }
    }

    Ok(result)
}

#[async_trait]
impl LabelStore for CachedDirectoryLabelStore {
    async fn labels(&self) -> io::Result<Vec<Label>> {
        let labels = self.labels.read().await;
        Ok(labels.values().cloned().collect())
    }

    async fn create_label(&self, label: &str) -> io::Result<Label> {
        let mut labels = self.labels.write().await;
        if labels.contains_key(label) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "database already exists",
            ));
        }

        let mut p = self.path.clone();
        p.push(format!("{}.label", label));
        let contents = b"0\n\n";
        match fs::metadata(&p).await {
            Ok(_) => Err(io::Error::new(
                io::ErrorKind::Other,
                "label was not in cached map but was found on disk",
            )),
            Err(e) => match e.kind() {
                io::ErrorKind::NotFound => {
                    let mut options = fs::OpenOptions::new();
                    options.create_new(true);
                    options.write(true);
                    let mut file = options.open(p).await?;
                    file.write_all(contents).await?;
                    file.flush().await?;
                    file.sync_all().await?;

                    let l = Label::new_empty(label);
                    labels.insert(label.to_string(), l.clone());

                    Ok(l)
                }
                _ => Err(e),
            },
        }
    }
    async fn get_label(&self, label: &str) -> io::Result<Option<Label>> {
        let labels = self.labels.read().await;
        Ok(labels.get(label).cloned())
    }
    async fn set_label_option(
        &self,
        label: &Label,
        layer: Option<[u32; 5]>,
    ) -> io::Result<Option<Label>> {
        let new_label = label.with_updated_layer(layer);
        let contents = match new_label.layer {
            None => format!("{}\n\n", new_label.version).into_bytes(),
            Some(layer) => {
                format!("{}\n{}\n", new_label.version, layer::name_to_string(layer)).into_bytes()
            }
        };

        let mut labels = self.labels.write().await;
        if let Some(retrieved_label) = labels.get(&label.name) {
            if retrieved_label == label {
                // all good, let's a go
                let mut p = self.path.clone();
                p.push(format!("{}.label", label.name));
                let mut options = fs::OpenOptions::new();
                options.create(false);
                options.write(true);
                let mut file = options.open(p).await?;
                file.write_all(&contents).await?;
                file.flush().await?;
                file.sync_data().await?;

                labels.insert(label.name.clone(), new_label.clone());
                Ok(Some(new_label))
            } else {
                Ok(None)
            }
        } else {
            Err(io::Error::new(io::ErrorKind::NotFound, "label not found"))
        }
    }

    async fn delete_label(&self, name: &str) -> io::Result<bool> {
        let mut labels = self.labels.write().await;
        if labels.remove(name).is_some() {
            let mut p = self.path.clone();
            p.push(format!("{}.label", name));
            tokio::fs::remove_file(p).await?;

            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::*;
    use std::process::{Command, Stdio};
    use tempfile::tempdir;

    const PROCESS_ROOT: &str = "TERMINUS_STORE_DIRECTORY_PROCESS_ROOT";
    const PROCESS_ID: &str = "TERMINUS_STORE_DIRECTORY_PROCESS_ID";
    const PROCESS_IDS: &str = "TERMINUS_STORE_DIRECTORY_PROCESS_IDS";
    const PROCESS_PAYLOAD: &str = "TERMINUS_STORE_DIRECTORY_PROCESS_PAYLOAD";

    #[tokio::test]
    async fn directory_publication_process_child() {
        let Ok(root) = std::env::var(PROCESS_ROOT) else {
            return;
        };
        let ids = match std::env::var(PROCESS_IDS) {
            Ok(value) => value
                .split(',')
                .map(string_to_name)
                .collect::<io::Result<Vec<_>>>()
                .unwrap(),
            Err(_) => vec![string_to_name(&std::env::var(PROCESS_ID).unwrap()).unwrap()],
        };
        let payload = std::env::var(PROCESS_PAYLOAD).unwrap().into_bytes();
        let multiple = ids.len() > 1;
        let layers = ids
            .into_iter()
            .enumerate()
            .map(|(index, id)| {
                let mut files = HashMap::new();
                let mut bytes = payload.clone();
                if multiple {
                    bytes.extend_from_slice(index.to_string().as_bytes());
                }
                files.insert(FILENAMES.proof_state.to_string(), bytes);
                (id, files)
            })
            .collect();
        let store = DirectoryLayerStore::new(root);
        let result = store.publish_imported_layers(layers).await;
        println!(
            "PROCESS_RESULT={}",
            match result {
                Ok(()) => "won",
                Err(ref error) if is_indeterminate_import_commit(error) => "indeterminate",
                Err(_) => "lost",
            }
        );
    }

    #[tokio::test]
    async fn directory_recovery_process_child() {
        let Ok(root) = std::env::var(PROCESS_ROOT) else {
            return;
        };
        let store = DirectoryLayerStore::new(root);
        let _ = store.directories().await;
    }

    fn publication_child(
        root: &std::path::Path,
        id: [u32; 5],
        payload: &str,
        crash: Option<&str>,
    ) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("storage::directory::tests::directory_publication_process_child")
            .arg("--nocapture")
            .env(PROCESS_ROOT, root)
            .env(PROCESS_ID, name_to_string(id))
            .env(PROCESS_PAYLOAD, payload)
            .env("RUST_TEST_THREADS", "1")
            .env_remove("TERMINUS_STORE_DIRECTORY_CRASH_POINT")
            .env_remove("TERMINUS_STORE_DIRECTORY_FAIL_POST_COMMIT_FSYNC")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(point) = crash {
            command.env("TERMINUS_STORE_DIRECTORY_CRASH_POINT", point);
        }
        command
    }

    fn batch_publication_child(
        root: &std::path::Path,
        ids: &[[u32; 5]],
        payload: &str,
        crash: Option<&str>,
    ) -> Command {
        let mut command = publication_child(root, ids[0], payload, crash);
        command.env(
            PROCESS_IDS,
            ids.iter()
                .map(|id| name_to_string(*id))
                .collect::<Vec<_>>()
                .join(","),
        );
        command
    }

    fn recovery_child(root: &std::path::Path, crash: Option<&str>) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("storage::directory::tests::directory_recovery_process_child")
            .arg("--nocapture")
            .env(PROCESS_ROOT, root)
            .env("RUST_TEST_THREADS", "1")
            .env_remove("TERMINUS_STORE_DIRECTORY_CRASH_POINT")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(point) = crash {
            command.env("TERMINUS_STORE_DIRECTORY_CRASH_POINT", point);
        }
        command
    }

    #[tokio::test]
    async fn two_process_directory_publication_has_one_winner() {
        let root = tempdir().unwrap();
        let id = [91, 92, 93, 94, 95];
        let first = publication_child(root.path(), id, "first", None)
            .spawn()
            .unwrap();
        let second = publication_child(root.path(), id, "second", None)
            .spawn()
            .unwrap();
        let first = first.wait_with_output().unwrap();
        let second = second.wait_with_output().unwrap();
        assert!(first.status.success());
        assert!(second.status.success());
        let outputs = [first.stdout, second.stdout];
        assert_eq!(
            outputs
                .iter()
                .filter(|bytes| String::from_utf8_lossy(bytes).contains("PROCESS_RESULT=won"))
                .count(),
            1
        );
        let store = DirectoryLayerStore::new(root.path());
        assert!(store.directory_exists(id).await.unwrap());
        let bytes = store
            .get_file(id, FILENAMES.proof_state)
            .await
            .unwrap()
            .map()
            .await
            .unwrap();
        assert!(bytes.as_ref() == b"first" || bytes.as_ref() == b"second");
    }

    #[tokio::test]
    async fn directory_crash_points_recover_to_atomic_visibility() {
        for (point, committed) in [
            ("after_staging_fsync", false),
            ("after_target_rename", false),
            ("after_commit_marker_rename", true),
        ] {
            let root = tempdir().unwrap();
            let id = [101, point.len() as u32, 103, 104, 105];
            let output = publication_child(root.path(), id, "payload", Some(point))
                .output()
                .unwrap();
            assert!(
                !output.status.success(),
                "crash hook {} did not terminate child",
                point
            );
            let store = DirectoryLayerStore::new(root.path());
            let layers = store.directories().await.unwrap();
            assert_eq!(layers.contains(&id), committed, "crash point {point}");
            assert_eq!(store.directory_exists(id).await.unwrap(), committed);
        }
    }

    #[tokio::test]
    async fn multi_layer_process_crashes_are_all_or_none_to_readers() {
        let ids = [
            [201, 202, 203, 204, 205],
            [211, 212, 213, 214, 215],
            [221, 222, 223, 224, 225],
        ];
        for (point, committed) in [
            ("after_staging_fsync", false),
            ("after_target_rename", false),
            ("after_commit_marker_rename", true),
        ] {
            let root = tempdir().unwrap();
            let output = batch_publication_child(root.path(), &ids, "batch", Some(point))
                .output()
                .unwrap();
            assert!(
                !output.status.success(),
                "crash hook {} did not fire",
                point
            );
            let store = DirectoryLayerStore::new(root.path());
            let visible_before_recovery =
                futures::future::try_join_all(ids.iter().map(|id| store.directory_exists(*id)))
                    .await
                    .unwrap();
            let visible_count = visible_before_recovery
                .iter()
                .filter(|visible| **visible)
                .count();
            assert_eq!(visible_count, if committed { ids.len() } else { 0 });
            let listed = store.directories().await.unwrap();
            assert_eq!(
                ids.iter().filter(|id| listed.contains(id)).count(),
                if committed { ids.len() } else { 0 }
            );
        }
    }

    #[tokio::test]
    async fn overlapping_multi_layer_batches_have_one_cross_process_winner() {
        let root = tempdir().unwrap();
        let first_ids = [[231, 232, 233, 234, 235], [241, 242, 243, 244, 245]];
        let second_ids = [[241, 242, 243, 244, 245], [251, 252, 253, 254, 255]];
        let first = batch_publication_child(root.path(), &first_ids, "first", None)
            .spawn()
            .unwrap();
        let second = batch_publication_child(root.path(), &second_ids, "second", None)
            .spawn()
            .unwrap();
        let first = first.wait_with_output().unwrap();
        let second = second.wait_with_output().unwrap();
        assert!(first.status.success());
        assert!(second.status.success());
        let outputs = [first.stdout, second.stdout];
        assert_eq!(
            outputs
                .iter()
                .filter(|bytes| String::from_utf8_lossy(bytes).contains("PROCESS_RESULT=won"))
                .count(),
            1
        );
        let store = DirectoryLayerStore::new(root.path());
        let listed = store.directories().await.unwrap();
        assert!(listed.contains(&first_ids[1]));
        assert_eq!(
            listed.contains(&first_ids[0]),
            !listed.contains(&second_ids[1])
        );
        assert_eq!(listed.len(), 2);
    }

    #[tokio::test]
    async fn crash_during_quarantine_is_recovered_without_visible_residue() {
        let root = tempdir().unwrap();
        let ids = [[261, 262, 263, 264, 265], [271, 272, 273, 274, 275]];
        let publication =
            batch_publication_child(root.path(), &ids, "batch", Some("after_target_rename"))
                .output()
                .unwrap();
        assert!(!publication.status.success());
        let recovery = recovery_child(root.path(), Some("during_quarantine"))
            .output()
            .unwrap();
        assert!(!recovery.status.success());

        let store = DirectoryLayerStore::new(root.path());
        assert!(store.directories().await.unwrap().is_empty());
        for prefix in std::fs::read_dir(root.path()).unwrap() {
            let prefix = prefix.unwrap();
            if !prefix.file_type().unwrap().is_dir() {
                continue;
            }
            for entry in std::fs::read_dir(prefix.path()).unwrap() {
                let name = entry.unwrap().file_name().to_string_lossy().into_owned();
                assert!(!name.contains(".recovery-") && !name.contains(".import-"));
            }
        }
    }

    #[tokio::test]
    async fn post_commit_fsync_failure_is_explicitly_indeterminate() {
        let root = tempdir().unwrap();
        let id = [111, 112, 113, 114, 115];
        let output = publication_child(root.path(), id, "payload", None)
            .env("TERMINUS_STORE_DIRECTORY_FAIL_POST_COMMIT_FSYNC", "1")
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("PROCESS_RESULT=indeterminate"));
        let store = DirectoryLayerStore::new(root.path());
        assert!(store.directory_exists(id).await.unwrap());
    }

    #[tokio::test]
    async fn write_and_read_file_backed() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("foo");
        let file = FileBackedStore::new(file_path);

        let mut w = file.open_write().await.unwrap();
        w.write_all(&[1, 2, 3]).await.unwrap();
        w.flush().await.unwrap();
        let mut buf = Vec::new();
        file.open_read()
            .await
            .unwrap()
            .read_to_end(&mut buf)
            .await
            .unwrap();

        assert_eq!(vec![1, 2, 3], buf);
    }

    #[tokio::test]
    async fn write_and_map_file_backed() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("foo");
        let file = FileBackedStore::new(file_path);

        let mut w = file.open_write().await.unwrap();
        w.write_all(&[1, 2, 3]).await.unwrap();
        w.flush().await.unwrap();

        let map = file.map().await.unwrap();

        assert_eq!(&vec![1, 2, 3][..], &map.as_ref()[..]);
    }

    #[tokio::test]
    async fn write_and_map_large_file_backed() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("foo");
        let file = FileBackedStore::new(file_path);

        let mut w = file.open_write().await.unwrap();
        let mut contents = vec![0u8; 4096 << 4];
        for i in 0..contents.capacity() {
            contents[i] = (i as usize % 256) as u8;
        }

        w.write_all(&contents).await.unwrap();
        w.flush().await.unwrap();

        let map = file.map().await.unwrap();

        assert_eq!(contents, map.as_ref());
    }

    #[tokio::test]
    async fn create_layers_from_directory_store() {
        let dir = tempdir().unwrap();
        let store = DirectoryLayerStore::new(dir.path());

        let layer = async {
            let mut builder = store.create_base_layer().await?;
            let base_name = builder.name();

            builder.add_value_triple(ValueTriple::new_string_value("cow", "says", "moo"));
            builder.add_value_triple(ValueTriple::new_string_value("pig", "says", "oink"));
            builder.add_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));

            builder.commit_boxed().await?;

            let mut builder = store.create_child_layer(base_name).await?;
            let child_name = builder.name();

            builder.remove_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));
            builder.add_value_triple(ValueTriple::new_node("cow", "likes", "pig"));

            builder.commit_boxed().await?;

            store.get_layer(child_name).await
        }
        .await
        .unwrap()
        .unwrap();

        assert!(layer.value_triple_exists(&ValueTriple::new_string_value("cow", "says", "moo")));
        assert!(layer.value_triple_exists(&ValueTriple::new_string_value("pig", "says", "oink")));
        assert!(layer.value_triple_exists(&ValueTriple::new_node("cow", "likes", "pig")));
        assert!(!layer.value_triple_exists(&ValueTriple::new_string_value("duck", "says", "quack")));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn proof_state_publication_is_atomic_and_write_once() {
        let dir = tempdir().unwrap();
        let store = DirectoryLayerStore::new(dir.path());
        let mut builder = store.create_base_layer().await.unwrap();
        let name = builder.name();
        builder.add_value_triple(ValueTriple::new_node("a", "p", "b"));
        builder.commit_boxed().await.unwrap();

        let first_store = store.clone();
        let second_store = store.clone();
        let (first, second) = tokio::join!(
            first_store.prepare_proof_layer(name),
            second_store.prepare_proof_layer(name)
        );
        assert_ne!(first.is_ok(), second.is_ok());
        let stored = store.proof_layer(name).await.unwrap();
        store
            .validate_proof_layer_against_store(name)
            .await
            .unwrap();
        let winner = first.ok().or_else(|| second.ok()).unwrap();
        assert_eq!(stored.state, winner.state);
    }

    fn imported(id: [u32; 5]) -> ([u32; 5], HashMap<String, Vec<u8>>) {
        (id, HashMap::new())
    }

    #[tokio::test]
    async fn failed_directory_batch_collision_publishes_nothing_new() {
        let dir = tempdir().unwrap();
        let store = DirectoryLayerStore::new(dir.path());
        let first = [1, 2, 3, 4, 5];
        let collision = [6, 7, 8, 9, 10];
        store.create_named_directory(collision).await.unwrap();

        let error = store
            .publish_imported_layers(vec![imported(first), imported(collision)])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert!(!store.directory_exists(first).await.unwrap());
        assert!(store.directory_exists(collision).await.unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn overlapping_directory_batches_have_one_atomic_winner() {
        let dir = tempdir().unwrap();
        let first_store = DirectoryLayerStore::new(dir.path());
        let second_store = DirectoryLayerStore::new(dir.path());
        let first = [11, 12, 13, 14, 15];
        let overlap = [16, 17, 18, 19, 20];
        let second = [21, 22, 23, 24, 25];

        let (left, right) = tokio::join!(
            first_store.publish_imported_layers(vec![imported(first), imported(overlap)]),
            second_store.publish_imported_layers(vec![imported(overlap), imported(second)])
        );
        assert_ne!(left.is_ok(), right.is_ok());
        let first_visible = first_store.directory_exists(first).await.unwrap();
        let overlap_visible = first_store.directory_exists(overlap).await.unwrap();
        let second_visible = first_store.directory_exists(second).await.unwrap();
        assert!(overlap_visible);
        assert!(
            (first_visible && !second_visible) || (!first_visible && second_visible),
            "readers must observe exactly one complete overlapping transaction"
        );
    }

    #[tokio::test]
    async fn reopen_recovers_uncommitted_directory_target_and_preserves_commit() {
        let dir = tempdir().unwrap();
        let store = DirectoryLayerStore::new(dir.path());
        let abandoned = [26, 27, 28, 29, 30];
        let token = name_to_string([31, 32, 33, 34, 35]);
        let abandoned_path = store.layer_path(abandoned);
        fs::create_dir_all(&abandoned_path).await.unwrap();
        fs::write(abandoned_path.join(IMPORT_MARKER), token)
            .await
            .unwrap();
        assert!(!store.directory_exists(abandoned).await.unwrap());

        let reopened = DirectoryLayerStore::new(dir.path());
        assert!(!reopened.directories().await.unwrap().contains(&abandoned));
        assert!(matches!(
            fs::metadata(&abandoned_path).await,
            Err(error) if error.kind() == io::ErrorKind::NotFound
        ));

        let committed = [36, 37, 38, 39, 40];
        reopened
            .publish_imported_layers(vec![imported(committed)])
            .await
            .unwrap();
        let reopened_again = DirectoryLayerStore::new(dir.path());
        assert!(reopened_again.directory_exists(committed).await.unwrap());
        assert!(reopened_again
            .directories()
            .await
            .unwrap()
            .contains(&committed));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn directory_recovery_and_normal_create_are_serialized() {
        let dir = tempdir().unwrap();
        let store = DirectoryLayerStore::new(dir.path());
        let id = [41, 42, 43, 44, 45];
        let target = store.layer_path(id);
        fs::create_dir_all(&target).await.unwrap();
        fs::write(
            target.join(IMPORT_MARKER),
            name_to_string([46, 47, 48, 49, 50]),
        )
        .await
        .unwrap();

        let recovery = store.clone();
        let writer = store.clone();
        let (recovered, created) =
            tokio::join!(recovery.directories(), writer.create_named_directory(id));
        recovered.unwrap();
        created.unwrap();
        assert!(store.directory_exists(id).await.unwrap());
        assert!(!store.file_exists(id, IMPORT_MARKER).await.unwrap());
    }

    #[tokio::test]
    async fn directory_create_and_retrieve_equal_label() {
        let dir = tempdir().unwrap();
        let store = DirectoryLabelStore::new(dir.path());

        let (stored, retrieved) = async {
            let stored = store.create_label("foo").await?;
            let retrieved = store.get_label("foo").await?;

            Ok::<_, io::Error>((stored, retrieved))
        }
        .await
        .unwrap();

        assert_eq!(None, stored.layer);
        assert_eq!(stored, retrieved.unwrap());
    }

    #[tokio::test]
    async fn directory_update_label_succeeds() {
        let dir = tempdir().unwrap();
        let store = DirectoryLabelStore::new(dir.path());

        let retrieved = async {
            let stored = store.create_label("foo").await?;
            store.set_label(&stored, [6, 7, 8, 9, 10]).await?;

            store.get_label("foo").await
        }
        .await
        .unwrap()
        .unwrap();

        assert_eq!(Some([6, 7, 8, 9, 10]), retrieved.layer);
    }

    #[tokio::test]
    async fn directory_update_label_twice_from_same_label_object_fails() {
        let dir = tempdir().unwrap();
        let store = DirectoryLabelStore::new(dir.path());

        let (stored2, stored3) = async {
            let stored1 = store.create_label("foo").await?;

            let stored2 = store.set_label(&stored1, [6, 7, 8, 9, 10]).await?;
            let stored3 = store.set_label(&stored1, [10, 9, 8, 7, 6]).await?;

            Ok::<_, io::Error>((stored2, stored3))
        }
        .await
        .unwrap();

        assert!(stored2.is_some());
        assert!(stored3.is_none());
    }

    #[tokio::test]
    async fn directory_create_label_twice_errors() {
        let dir = tempdir().unwrap();
        let store = DirectoryLabelStore::new(dir.path());

        store.create_label("foo").await.unwrap();
        let result = store.create_label("foo").await;

        assert!(result.is_err());

        let error = result.err().unwrap();
        assert_eq!(io::ErrorKind::InvalidInput, error.kind());
    }

    #[tokio::test]
    async fn nonexistent_file_is_nonexistent() {
        let file = FileBackedStore::new("asdfasfopivbuzxcvopiuvpoawehkafpouzvxv");
        assert!(!file.exists().await.unwrap());
    }

    #[tokio::test]
    async fn rollup_and_retrieve_base() {
        let dir = tempdir().unwrap();
        let store = Arc::new(DirectoryLayerStore::new(dir.path()));

        let mut builder = store.create_base_layer().await.unwrap();
        let base_name = builder.name();

        builder.add_value_triple(ValueTriple::new_string_value("cow", "says", "moo"));
        builder.add_value_triple(ValueTriple::new_string_value("pig", "says", "oink"));
        builder.add_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));

        builder.commit_boxed().await.unwrap();

        let mut builder = store.create_child_layer(base_name).await.unwrap();
        let child_name = builder.name();

        builder.remove_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));
        builder.add_value_triple(ValueTriple::new_node("cow", "likes", "pig"));

        builder.commit_boxed().await.unwrap();

        let unrolled_layer = store.get_layer(child_name).await.unwrap().unwrap();

        // This stack is built directly through the low-level builder API, which does
        // not attach the per-layer intrinsic commitment (`proof_state`) that the
        // intrinsic-commitment feature (commit cbe48ac) now requires on the rollup
        // path. Run the intended one-time proof-chain migration over the stack so the
        // rollup below can read each ancestor's commitment instead of failing with
        // "run migrate_proof_chain once". Migration only writes proof metadata; it
        // does not alter the layers' triples, so the assertions below are unchanged.
        store.migrate_proof_chain(child_name).await.unwrap();

        let _rolled_id = store.clone().rollup(unrolled_layer).await.unwrap();
        let rolled_layer = store.get_layer(child_name).await.unwrap().unwrap();

        match *rolled_layer {
            InternalLayer::Rollup(_) => {}
            _ => panic!("not a rollup"),
        }

        assert!(
            rolled_layer.value_triple_exists(&ValueTriple::new_string_value("cow", "says", "moo"))
        );
        assert!(
            rolled_layer.value_triple_exists(&ValueTriple::new_string_value("pig", "says", "oink"))
        );
        assert!(rolled_layer.value_triple_exists(&ValueTriple::new_node("cow", "likes", "pig")));
        assert!(!rolled_layer
            .value_triple_exists(&ValueTriple::new_string_value("duck", "says", "quack")));
    }

    #[tokio::test]
    async fn rollup_and_retrieve_child() {
        let dir = tempdir().unwrap();
        let store = Arc::new(DirectoryLayerStore::new(dir.path()));

        let mut builder = store.create_base_layer().await.unwrap();
        let base_name = builder.name();

        builder.add_value_triple(ValueTriple::new_string_value("cow", "says", "moo"));
        builder.add_value_triple(ValueTriple::new_string_value("pig", "says", "oink"));
        builder.add_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));

        builder.commit_boxed().await.unwrap();

        let mut builder = store.create_child_layer(base_name).await.unwrap();
        let child_name = builder.name();

        builder.remove_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));
        builder.add_value_triple(ValueTriple::new_node("cow", "likes", "pig"));

        builder.commit_boxed().await.unwrap();

        let mut builder = store.create_child_layer(child_name).await.unwrap();
        let child_name = builder.name();

        builder.remove_value_triple(ValueTriple::new_string_value("cow", "likes", "pig"));
        builder.add_value_triple(ValueTriple::new_node("cow", "hates", "pig"));

        builder.commit_boxed().await.unwrap();

        let unrolled_layer = store.get_layer(child_name).await.unwrap().unwrap();

        // Legacy stack built via the low-level builder API (no per-layer intrinsic
        // commitment). Run the one-time proof-chain migration on the head so the
        // rollup below can read each ancestor's `proof_state` instead of failing with
        // "run migrate_proof_chain once". Adds proof metadata only; triples (and thus
        // the assertions below) are unchanged. `child_name` here is the head
        // (grandchild) after the earlier shadowing, so the whole chain is migrated.
        store.migrate_proof_chain(child_name).await.unwrap();

        let _rolled_id = store
            .clone()
            .rollup_upto(unrolled_layer, base_name)
            .await
            .unwrap();
        let rolled_layer = store.get_layer(child_name).await.unwrap().unwrap();

        match *rolled_layer {
            InternalLayer::Rollup(_) => {}
            _ => panic!("not a rollup"),
        }

        assert!(
            rolled_layer.value_triple_exists(&ValueTriple::new_string_value("cow", "says", "moo"))
        );
        assert!(
            rolled_layer.value_triple_exists(&ValueTriple::new_string_value("pig", "says", "oink"))
        );
        assert!(rolled_layer.value_triple_exists(&ValueTriple::new_node("cow", "hates", "pig")));
        assert!(!rolled_layer
            .value_triple_exists(&ValueTriple::new_string_value("cow", "likes", "pig")));
        assert!(!rolled_layer
            .value_triple_exists(&ValueTriple::new_string_value("duck", "says", "quack")));
    }

    #[tokio::test]
    async fn create_and_delete_label() {
        let dir = tempdir().unwrap();
        let store = DirectoryLabelStore::new(dir.path());

        store.create_label("foo").await.unwrap();
        assert!(store.get_label("foo").await.unwrap().is_some());
        assert!(store.delete_label("foo").await.unwrap());
        assert!(store.get_label("foo").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_nonexistent_label() {
        let dir = tempdir().unwrap();
        let store = DirectoryLabelStore::new(dir.path());

        assert!(!store.delete_label("foo").await.unwrap());
    }

    #[tokio::test]
    async fn delete_shared_locked_label() {
        let dir = tempdir().unwrap();
        let store = DirectoryLabelStore::new(dir.path());

        store.create_label("foo").await.unwrap();
        let label_path = dir.path().join("foo.label");
        let _f = LockedFile::open(label_path).await.unwrap();

        assert!(store.delete_label("foo").await.unwrap());
    }

    #[tokio::test]
    async fn delete_exclusive_locked_label() {
        let dir = tempdir().unwrap();
        let store = DirectoryLabelStore::new(dir.path());

        store.create_label("foo").await.unwrap();
        let label_path = dir.path().join("foo.label");
        let _f = ExclusiveLockedFile::open(label_path).await.unwrap();

        assert!(store.delete_label("foo").await.unwrap());
    }
}
