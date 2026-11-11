// File format:
// <header>
//  [<filetype present>]*
//  [<offsets>]*
//

use std::{
    collections::HashMap,
    convert::TryFrom,
    io::{self, ErrorKind, SeekFrom},
    ops::Range,
    path::PathBuf,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
    task::Poll,
};

#[cfg(not(target_os = "windows"))]
use std::os::unix::fs::MetadataExt;
#[cfg(target_os = "windows")]
use std::os::windows::fs::MetadataExt;

use async_trait::async_trait;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use fs2::FileExt as Fs2FileExt;
use lru::LruCache;
use num_traits::FromPrimitive;
use tokio::{
    fs::{self, File},
    io::{AsyncRead, AsyncReadExt, AsyncSeekExt, AsyncWrite, AsyncWriteExt},
};
use tokio_util::either::Either;

use tdb_succinct::{
    logarray_length_from_control_word, smallbitarray::SmallBitArray, LateLogArrayBufBuilder,
    MonotonicLogArray,
};

use super::{
    consts::{LayerFileEnum, FILENAMES, FILENAME_ENUM_MAP},
    indeterminate_import_commit_error,
    locking::{ExclusiveLockedFile, LockedFile},
    name_to_string, string_to_name, FileLoad, FileStore, PersistentLayerStore, SyncableFile,
};

#[cfg(test)]
use super::is_indeterminate_import_commit;

const PREFIX_DIR_SIZE: usize = 3;
const ARCHIVE_IMPORT_MARKER_SUFFIX: &str = "larch.import-tx";
const IMPORT_COMMIT_DIR: &str = ".terminus-store-import-commits";
const IMPORT_LOCK: &str = ".terminus-store-import.lock";

#[cfg(test)]
fn injected_crash(point: &str) {
    if std::env::var("TERMINUS_STORE_ARCHIVE_CRASH_POINT").as_deref() == Ok(point) {
        std::process::abort();
    }
}

#[async_trait]
pub trait ArchiveBackend: Clone + Send + Sync {
    type Read: AsyncRead + Unpin + Send;
    async fn get_layer_bytes(&self, id: [u32; 5]) -> io::Result<Bytes>;
    async fn get_layer_structure_bytes(
        &self,
        id: [u32; 5],
        file_type: LayerFileEnum,
    ) -> io::Result<Option<Bytes>>;
    /// True only when the entire transaction containing this layer is visible.
    async fn layer_is_visible(&self, id: [u32; 5]) -> io::Result<bool>;
    async fn store_layer_file(&self, id: [u32; 5], bytes: Bytes) -> io::Result<()>;
    async fn store_layer_file_if_absent(&self, id: [u32; 5], bytes: Bytes) -> io::Result<()>;
    /// Atomically replace exactly the bytes read by a migration, or fail if a
    /// concurrent writer changed the archive.
    async fn replace_layer_file_if_matches(
        &self,
        id: [u32; 5],
        expected: Bytes,
        replacement: Bytes,
    ) -> io::Result<()>;
    /// Publish a validated multi-layer batch with one visibility commit point.
    async fn publish_layer_files_atomically(
        &self,
        layers: Vec<([u32; 5], Bytes)>,
    ) -> io::Result<()>;
    async fn remove_layer_file(&self, id: [u32; 5]) -> io::Result<()>;
    async fn read_layer_structure_bytes_from(
        &self,
        id: [u32; 5],
        file_type: LayerFileEnum,
        read_from: usize,
    ) -> io::Result<Self::Read>;
}

#[async_trait]
pub trait ArchiveMetadataBackend: Clone + Send + Sync {
    async fn get_layer_names(&self) -> io::Result<Vec<[u32; 5]>>;
    async fn layer_exists(&self, id: [u32; 5]) -> io::Result<bool>;
    async fn layer_size(&self, id: [u32; 5]) -> io::Result<u64>;
    async fn layer_file_exists(&self, id: [u32; 5], file_type: LayerFileEnum) -> io::Result<bool>;
    async fn get_layer_structure_size(
        &self,
        id: [u32; 5],
        file_type: LayerFileEnum,
    ) -> io::Result<usize>;
    async fn get_rollup(&self, id: [u32; 5]) -> io::Result<Option<[u32; 5]>>;
    async fn set_rollup(&self, id: [u32; 5], rollup: [u32; 5]) -> io::Result<()>;
    /// The rollup a layer is registered to, with the registration's version
    /// line. Version 2 marks a rollup this store validated as resolving every
    /// id of the original layer to the same term (see
    /// [`crate::storage::layer::VALIDATED_ROLLUP_VERSION`]).
    async fn get_rollup_record(&self, id: [u32; 5]) -> io::Result<Option<(u32, [u32; 5])>> {
        Ok(self.get_rollup(id).await?.map(|rollup| (1, rollup)))
    }
    async fn set_rollup_record(
        &self,
        id: [u32; 5],
        rollup: [u32; 5],
        _version: u32,
    ) -> io::Result<()> {
        self.set_rollup(id, rollup).await
    }
    async fn get_parent(&self, id: [u32; 5]) -> io::Result<Option<[u32; 5]>>;
}

pub struct BytesAsyncReader(Bytes);

impl AsyncRead for BytesAsyncReader {
    fn poll_read(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let bytes = &mut self.get_mut().0;
        let consumed = if buf.remaining() > bytes.len() {
            bytes.split_to(bytes.len())
        } else {
            bytes.split_to(buf.remaining())
        };

        buf.put_slice(consumed.as_ref());

        Poll::Ready(Ok(()))
    }
}

#[derive(Clone)]
pub struct DirectoryArchiveBackend {
    path: PathBuf,
}

impl DirectoryArchiveBackend {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
    fn path_for_layer(&self, name: [u32; 5]) -> PathBuf {
        let mut p = self.path.clone();
        let name_str = name_to_string(name);
        p.push(&name_str[0..PREFIX_DIR_SIZE]);
        p.push(&format!("{}.larch", name_str));

        p
    }

    fn path_for_rollup(&self, name: [u32; 5]) -> PathBuf {
        let mut p = self.path.clone();
        let name_str = name_to_string(name);
        p.push(&name_str[0..PREFIX_DIR_SIZE]);
        p.push(&format!("{}.rollup.hex", name_str));

        p
    }

    fn path_for_import_marker(&self, name: [u32; 5]) -> PathBuf {
        self.path_for_layer(name)
            .with_extension(ARCHIVE_IMPORT_MARKER_SUFFIX)
    }

    fn path_for_write_lock(&self, name: [u32; 5]) -> PathBuf {
        self.path_for_layer(name).with_extension("larch.write-lock")
    }

    fn import_commit_path(&self, token: &str) -> io::Result<PathBuf> {
        string_to_name(token)?;
        Ok(self
            .path
            .join(IMPORT_COMMIT_DIR)
            .join(format!("{token}.commit")))
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

    async fn ensure_prefix(&self, id: [u32; 5]) -> io::Result<PathBuf> {
        fs::create_dir_all(&self.path).await?;
        let prefix = self.path_for_layer(id).parent().unwrap().to_path_buf();
        Self::ensure_child_directory(&self.path, &prefix).await?;
        Ok(prefix)
    }

    async fn acquire_lock_path(path: PathBuf) -> io::Result<std::fs::File> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        tokio::task::spawn_blocking(move || {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(path)?;
            Fs2FileExt::lock_exclusive(&file)?;
            Ok(file)
        })
        .await
        .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?
    }

    async fn acquire_import_lock(&self) -> io::Result<std::fs::File> {
        let file = Self::acquire_lock_path(self.path.join(IMPORT_LOCK)).await?;
        Self::sync_directory(&self.path).await?;
        Ok(file)
    }

    async fn acquire_layer_lock(&self, id: [u32; 5]) -> io::Result<std::fs::File> {
        let prefix = self.ensure_prefix(id).await?;
        let file = Self::acquire_lock_path(self.path_for_write_lock(id)).await?;
        Self::sync_directory(&prefix).await?;
        Ok(file)
    }

    async fn quarantine_import_locked(
        &self,
        id: [u32; 5],
        marker: &std::path::Path,
    ) -> io::Result<()> {
        let layer = self.path_for_layer(id);
        let parent = layer.parent().unwrap().to_path_buf();
        let name = layer.file_name().unwrap().to_string_lossy();
        let trash = parent.join(format!(".{name}.recovery-{:016x}", rand::random::<u64>()));
        let moved = match rename_archive_noreplace(layer, trash.clone()).await {
            Ok(()) => {
                Self::sync_directory(&parent).await?;
                #[cfg(test)]
                injected_crash("during_quarantine");
                true
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => false,
            Err(error) => return Err(error),
        };
        // It is now safe to remove the visibility marker: no target remains at
        // the visible name. Persist both removals before deleting quarantine.
        match fs::remove_file(marker).await {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        Self::sync_directory(&parent).await?;
        if moved {
            fs::remove_file(&trash).await?;
            Self::sync_directory(&parent).await?;
        }
        Ok(())
    }

    async fn layer_visible(&self, id: [u32; 5]) -> io::Result<bool> {
        match fs::metadata(self.path_for_layer(id)).await {
            Ok(metadata) if metadata.is_file() => {}
            Ok(_) => return Ok(false),
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error),
        }
        let token = match fs::read_to_string(self.path_for_import_marker(id)).await {
            Ok(token) => token,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(true),
            Err(error) => return Err(error),
        };
        match fs::metadata(self.import_commit_path(token.trim())?).await {
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
                let name = entry.file_name();
                let Some(name) = name.to_str() else {
                    continue;
                };
                if name.starts_with('.')
                    && (name.contains(".import-")
                        || name.contains(".marker-")
                        || name.contains(".recovery-"))
                {
                    fs::remove_file(entry.path()).await?;
                    Self::sync_directory(&prefix.path()).await?;
                    continue;
                }
                if !name.ends_with(".larch.import-tx") || !entry.file_type().await?.is_file() {
                    continue;
                }
                let marker = entry.path();
                let token = fs::read_to_string(&marker).await?;
                let committed = match fs::metadata(self.import_commit_path(token.trim())?).await {
                    Ok(metadata) => metadata.is_file(),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => false,
                    Err(error) => return Err(error),
                };
                if !committed {
                    let id_text = name.strip_suffix(".larch.import-tx").ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "invalid archive import marker")
                    })?;
                    let id = string_to_name(id_text)?;
                    let _layer_lock = self.acquire_layer_lock(id).await?;
                    let current_token = match fs::read_to_string(&marker).await {
                        Ok(value) => value,
                        Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                        Err(error) => return Err(error),
                    };
                    let now_committed =
                        match fs::metadata(self.import_commit_path(current_token.trim())?).await {
                            Ok(metadata) => metadata.is_file(),
                            Err(error) if error.kind() == io::ErrorKind::NotFound => false,
                            Err(error) => return Err(error),
                        };
                    if !now_committed {
                        self.quarantine_import_locked(id, &marker).await?;
                    }
                }
            }
        }
        Ok(())
    }

    async fn write_layer_file(&self, id: [u32; 5], bytes: Bytes, replace: bool) -> io::Result<()> {
        let _lock = self.acquire_layer_lock(id).await?;
        self.write_layer_file_locked(id, bytes, replace).await
    }

    async fn write_layer_file_locked(
        &self,
        id: [u32; 5],
        mut bytes: Bytes,
        replace: bool,
    ) -> io::Result<()> {
        let path = self.path_for_layer(id);
        let directory_path = self.ensure_prefix(id).await?;
        let temporary_path =
            path.with_extension(format!("larch.tmp-{:016x}", rand::random::<u64>()));
        let result = async {
            let mut options = tokio::fs::OpenOptions::new();
            options.create_new(true).write(true);
            let mut file = options.open(&temporary_path).await.map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("open archive temporary file: {error}"),
                )
            })?;
            while bytes.remaining() > 0 {
                let chunk = bytes.chunk();
                let written = file.write(chunk).await?;
                bytes.advance(written);
            }
            file.flush().await?;
            file.sync_all().await?;
            if replace {
                fs::rename(&temporary_path, &path).await
            } else {
                rename_archive_noreplace(temporary_path.clone(), path.clone()).await
            }
            .map_err(|error| {
                io::Error::new(error.kind(), format!("publish archive file: {error}"))
            })?;
            if cfg!(unix) {
                fs::OpenOptions::new()
                    .read(true)
                    .open(&directory_path)
                    .await?
                    .sync_all()
                    .await?;
            }
            Ok::<_, io::Error>(())
        }
        .await;
        if result.is_err() {
            let _ = fs::remove_file(&temporary_path).await;
        }
        result
    }
}

#[cfg(target_os = "linux")]
async fn rename_archive_noreplace(source: PathBuf, target: PathBuf) -> io::Result<()> {
    tokio::task::spawn_blocking(move || {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let source = CString::new(source.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "nul in archive path"))?;
        let target = CString::new(target.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "nul in archive path"))?;
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
async fn rename_archive_noreplace(_source: PathBuf, _target: PathBuf) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace archive publication is unavailable on this platform",
    ))
}

#[async_trait]
impl ArchiveBackend for DirectoryArchiveBackend {
    type Read = ArchiveSliceReader;
    async fn get_layer_bytes(&self, id: [u32; 5]) -> io::Result<Bytes> {
        if !self.layer_visible(id).await? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "archive layer not found",
            ));
        }
        let path = self.path_for_layer(id);
        let mut options = fs::OpenOptions::new();
        options.read(true);
        options.create(false);
        let mut result = options.open(path).await?;
        let metadata = result.metadata().await?;
        #[cfg(target_os = "windows")]
        let size = metadata.file_size();
        #[cfg(not(target_os = "windows"))]
        let size = metadata.size();
        let mut buf = Vec::with_capacity(size as usize);
        result.read_to_end(&mut buf).await?;
        buf.shrink_to_fit();

        Ok(buf.into())
    }

    async fn get_layer_structure_bytes(
        &self,
        id: [u32; 5],
        file_type: LayerFileEnum,
    ) -> io::Result<Option<Bytes>> {
        if !self.layer_visible(id).await? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "archive layer not found",
            ));
        }
        let path = self.path_for_layer(id);
        let mut options = tokio::fs::OpenOptions::new();
        options.read(true);
        let mut file = options.open(path).await?;
        let header = ArchiveHeader::parse_from_reader(&mut file).await?;
        if let Some(range) = header.range_for(file_type) {
            let mut data = vec![0; range.len()];
            file.seek(SeekFrom::Current((range.start) as i64)).await?;
            file.read_exact(&mut data).await?;

            Ok(Some(Bytes::from(data)))
        } else {
            Ok(None)
        }
    }

    async fn layer_is_visible(&self, id: [u32; 5]) -> io::Result<bool> {
        self.layer_visible(id).await
    }

    async fn store_layer_file(&self, id: [u32; 5], bytes: Bytes) -> io::Result<()> {
        self.write_layer_file(id, bytes, true).await
    }

    async fn store_layer_file_if_absent(&self, id: [u32; 5], bytes: Bytes) -> io::Result<()> {
        self.write_layer_file(id, bytes, false).await
    }

    async fn replace_layer_file_if_matches(
        &self,
        id: [u32; 5],
        expected: Bytes,
        replacement: Bytes,
    ) -> io::Result<()> {
        let _lock = self.acquire_layer_lock(id).await?;
        let current = fs::read(self.path_for_layer(id)).await?;
        if current.as_slice() != expected.as_ref() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "archive layer changed during conditional replacement",
            ));
        }
        self.write_layer_file_locked(id, replacement, true).await
    }

    async fn publish_layer_files_atomically(
        &self,
        mut layers: Vec<([u32; 5], Bytes)>,
    ) -> io::Result<()> {
        let _import_lock = self.acquire_import_lock().await?;
        self.recover_uncommitted_imports_locked().await?;
        if layers.is_empty() {
            return Ok(());
        }
        layers.sort_by_key(|(id, _)| *id);
        if layers.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "duplicate imported archive layer id",
            ));
        }
        let mut layer_locks = Vec::with_capacity(layers.len());
        for (id, _) in &layers {
            layer_locks.push(self.acquire_layer_lock(*id).await?);
        }
        let token = name_to_string(rand::random());
        let commit_dir = self.path.join(IMPORT_COMMIT_DIR);
        Self::ensure_child_directory(&self.path, &commit_dir).await?;

        let mut staged = Vec::with_capacity(layers.len());
        for (id, bytes) in layers {
            let target = self.path_for_layer(id);
            let marker = self.path_for_import_marker(id);
            if fs::metadata(&target).await.is_ok() || fs::metadata(&marker).await.is_ok() {
                for (_, archive_temp, marker_temp, _, _) in &staged {
                    let _ = fs::remove_file(archive_temp).await;
                    let _ = fs::remove_file(marker_temp).await;
                }
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "archive import target already exists",
                ));
            }
            let parent = self.ensure_prefix(id).await?;
            let archive_temp = parent.join(format!(
                ".{}.import-{token}",
                target.file_name().unwrap().to_string_lossy()
            ));
            let marker_temp = parent.join(format!(
                ".{}.marker-{token}",
                target.file_name().unwrap().to_string_lossy()
            ));
            let stage_result = async {
                let mut archive_file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&archive_temp)
                    .await?;
                archive_file.write_all(&bytes).await?;
                archive_file.flush().await?;
                archive_file.sync_all().await?;
                let mut marker_file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&marker_temp)
                    .await?;
                marker_file.write_all(token.as_bytes()).await?;
                marker_file.flush().await?;
                marker_file.sync_all().await?;
                Ok::<_, io::Error>(())
            }
            .await;
            if let Err(error) = stage_result {
                let _ = fs::remove_file(&archive_temp).await;
                let _ = fs::remove_file(&marker_temp).await;
                for (_, archive_temp, marker_temp, _, _) in &staged {
                    let _ = fs::remove_file(archive_temp).await;
                    let _ = fs::remove_file(marker_temp).await;
                }
                return Err(error);
            }
            staged.push((id, archive_temp, marker_temp, target, marker));
        }

        #[cfg(test)]
        injected_crash("after_staging_fsync");

        let mut published: Vec<([u32; 5], PathBuf)> = Vec::new();
        for (id, archive_temp, marker_temp, target, marker) in &staged {
            let result = async {
                rename_archive_noreplace(marker_temp.clone(), marker.clone()).await?;
                #[cfg(test)]
                injected_crash("after_visibility_marker_rename");
                let parent = marker.parent().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "archive marker has no parent")
                })?;
                Self::sync_directory(parent).await?;
                #[cfg(test)]
                injected_crash("after_visibility_marker_fsync");
                rename_archive_noreplace(archive_temp.clone(), target.clone()).await?;
                Ok::<_, io::Error>(())
            }
            .await;
            if let Err(error) = result {
                for (published_id, marker) in &published {
                    let _ = self.quarantine_import_locked(*published_id, marker).await;
                }
                for (staged_id, archive_temp, marker_temp, target, marker) in &staged {
                    let _ = fs::remove_file(archive_temp).await;
                    let _ = fs::remove_file(marker_temp).await;
                    if fs::metadata(target).await.is_ok() {
                        let _ = self.quarantine_import_locked(*staged_id, marker).await;
                    } else {
                        let _ = fs::remove_file(marker).await;
                    }
                }
                return Err(error);
            }
            published.push((*id, marker.to_path_buf()));
            #[cfg(test)]
            injected_crash("after_target_rename");
        }
        let mut parents: Vec<_> = published
            .iter()
            .filter_map(|(id, _)| {
                self.path_for_layer(*id)
                    .parent()
                    .map(|parent| parent.to_path_buf())
            })
            .collect();
        parents.sort();
        parents.dedup();
        for parent in parents {
            Self::sync_directory(&parent).await?;
        }

        let commit = self.import_commit_path(&token)?;
        let commit_temp = commit_dir.join(format!(".{token}.tmp"));
        let mut commit_file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&commit_temp)
            .await?;
        commit_file.write_all(b"committed\n").await?;
        commit_file.flush().await?;
        commit_file.sync_all().await?;
        rename_archive_noreplace(commit_temp, commit).await?;
        #[cfg(test)]
        injected_crash("after_commit_marker_rename");
        // Logical visibility begins at the commit-marker rename, but success is
        // reported only after its parent directory is durable. A failure after
        // this point is explicitly typed as indeterminate so callers do not
        // mistake it for a retry-safe pre-commit failure.
        #[cfg(test)]
        let post_commit_durability =
            if std::env::var_os("TERMINUS_STORE_ARCHIVE_FAIL_POST_COMMIT_FSYNC").is_some() {
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
            .map_err(|error| indeterminate_import_commit_error("directory-archive", error))
    }

    async fn remove_layer_file(&self, id: [u32; 5]) -> io::Result<()> {
        let _lock = self.acquire_layer_lock(id).await?;
        let path = self.path_for_layer(id);
        let parent = path.parent().unwrap().to_path_buf();
        let result = match fs::remove_file(path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        };
        Self::sync_directory(&parent).await?;
        match fs::remove_file(self.path_for_import_marker(id)).await {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        Self::sync_directory(&parent).await?;
        result
    }

    async fn read_layer_structure_bytes_from(
        &self,
        id: [u32; 5],
        file_type: LayerFileEnum,
        read_from: usize,
    ) -> io::Result<Self::Read> {
        if !self.layer_visible(id).await? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "archive layer not found",
            ));
        }
        let path = self.path_for_layer(id);
        let mut options = tokio::fs::OpenOptions::new();
        options.read(true);
        let mut file = options.open(path).await?;
        let header = ArchiveHeader::parse_from_reader(&mut file).await?;

        let range = header
            .range_for(file_type)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "slice not found in archive"))?;

        let remaining = range.len() - read_from;
        file.seek(SeekFrom::Current((range.start + read_from) as i64))
            .await?;

        Ok(ArchiveSliceReader { file, remaining })
    }
}

#[async_trait]
impl ArchiveMetadataBackend for DirectoryArchiveBackend {
    async fn get_layer_names(&self) -> io::Result<Vec<[u32; 5]>> {
        let _import_lock = self.acquire_import_lock().await?;
        self.recover_uncommitted_imports_locked().await?;
        let mut prefixes = fs::read_dir(&self.path).await?;
        let mut result = Vec::new();
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
                let os_name = entry.file_name();
                let name = os_name.to_str().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unexpected non-utf8 archive layer name",
                    )
                })?;
                if name.ends_with(".larch") && entry.file_type().await?.is_file() {
                    let id = string_to_name(&name[..name.len() - 6])?;
                    if self.layer_visible(id).await? {
                        result.push(id);
                    }
                }
            }
        }

        Ok(result)
    }

    async fn layer_exists(&self, id: [u32; 5]) -> io::Result<bool> {
        self.layer_visible(id).await
    }

    async fn layer_size(&self, id: [u32; 5]) -> io::Result<u64> {
        if !self.layer_visible(id).await? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "archive layer not found",
            ));
        }
        let path = self.path_for_layer(id);
        let metadata = tokio::fs::metadata(path).await?;
        Ok(metadata.len())
    }

    async fn layer_file_exists(&self, id: [u32; 5], file_type: LayerFileEnum) -> io::Result<bool> {
        if !self.layer_visible(id).await? {
            return Ok(false);
        }
        let path = self.path_for_layer(id);
        let mut options = tokio::fs::OpenOptions::new();
        options.read(true);
        let mut file = options.open(path).await?;
        let header = ArchiveFilePresenceHeader::new(file.read_u64().await?);

        Ok(header.is_present(file_type))
    }

    async fn get_layer_structure_size(
        &self,
        id: [u32; 5],
        file_type: LayerFileEnum,
    ) -> io::Result<usize> {
        if !self.layer_visible(id).await? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "archive layer not found",
            ));
        }
        let path = self.path_for_layer(id);
        // read header!
        let mut options = tokio::fs::OpenOptions::new();
        options.read(true);
        let mut file = options.open(path).await?;
        let header = ArchiveHeader::parse_from_reader(&mut file).await?;

        header
            .size_of(file_type)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "slice not found in archive"))
    }

    async fn get_rollup(&self, id: [u32; 5]) -> io::Result<Option<[u32; 5]>> {
        Ok(self.get_rollup_record(id).await?.map(|(_, rollup)| rollup))
    }

    async fn get_rollup_record(&self, id: [u32; 5]) -> io::Result<Option<(u32, [u32; 5])>> {
        if !self.layer_visible(id).await? {
            return Ok(None);
        }
        // acquire a shared lock on the layer. This ensures nobody will write a rollup file while we're retrieving it.
        let layer_path = self.path_for_layer(id);
        let layer_lock = LockedFile::open(layer_path).await;
        if layer_lock.is_err() && layer_lock.as_ref().err().unwrap().kind() == ErrorKind::NotFound {
            // no such layer - therefore no such rollup
            return Ok(None);
        }
        let _layer_lock = layer_lock.unwrap();

        let path = self.path_for_rollup(id);
        let result = fs::read_to_string(path).await;

        if result.is_err() && result.as_ref().err().unwrap().kind() == ErrorKind::NotFound {
            return Ok(None);
        }
        let data = result?;
        let mut lines = data.lines();
        let version = lines.next().and_then(|line| line.parse::<u32>().ok()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "rollup file has no version line")
        })?;
        let name = lines.next().expect(
            "Expected rollup file to have two lines but was unable to skip to the second line",
        );
        Ok(Some((version, string_to_name(&name)?)))
    }

    async fn set_rollup(&self, id: [u32; 5], rollup: [u32; 5]) -> io::Result<()> {
        self.set_rollup_record(id, rollup, 1).await
    }

    async fn set_rollup_record(
        &self,
        id: [u32; 5],
        rollup: [u32; 5],
        version: u32,
    ) -> io::Result<()> {
        if !self.layer_visible(id).await? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "archive layer not found",
            ));
        }
        // acquire an exclusive lock on the layer. This ensures nobody tries to lookup the rollup while we're writing it.
        let layer_path = self.path_for_layer(id);
        let _layer_lock = ExclusiveLockedFile::open(layer_path).await?;

        let path = self.path_for_rollup(id);
        let mut data = Vec::with_capacity(43);
        data.extend_from_slice(format!("{version}\n").as_bytes());
        data.extend_from_slice(name_to_string(rollup).as_bytes());
        data.extend_from_slice(b"\n");
        let mut file = fs::File::create(path).await?;
        file.write_all(&data).await?;
        file.flush().await?;
        file.sync_all().await?;

        Ok(())
    }

    async fn get_parent(&self, id: [u32; 5]) -> io::Result<Option<[u32; 5]>> {
        if !self.layer_visible(id).await? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "archive layer not found",
            ));
        }
        if let Some(parent_bytes) = self
            .get_layer_structure_bytes(id, LayerFileEnum::Parent)
            .await?
        {
            let parent_string = std::str::from_utf8(&parent_bytes[..40]).unwrap();
            Ok(Some(string_to_name(parent_string).unwrap()))
        } else {
            Ok(None)
        }
    }
}

#[derive(Clone)]
pub struct LruArchiveBackend<M, D> {
    cache: Arc<tokio::sync::Mutex<LruCache<[u32; 5], CacheEntry>>>,
    #[cfg(feature = "rollup_metadata_experimental")]
    rollup_cache: Arc<std::sync::RwLock<HashMap<[u32; 5], Option<[u32; 5]>>>>,
    limit: usize,
    current: Arc<AtomicUsize>,
    metadata_origin: M,
    data_origin: D,
}

#[derive(Clone)]
enum CacheEntry {
    Resolving(Arc<tokio::sync::RwLock<Option<Result<Bytes, io::ErrorKind>>>>),
    Resolved(Bytes),
}

impl CacheEntry {
    fn is_resolving(&self) -> bool {
        if let Self::Resolving(_) = self {
            true
        } else {
            false
        }
    }
}

impl<M, D> LruArchiveBackend<M, D> {
    pub fn new(metadata_origin: M, data_origin: D, limit: usize) -> Self {
        let cache = Arc::new(tokio::sync::Mutex::new(LruCache::unbounded()));

        Self {
            cache,
            #[cfg(feature = "rollup_metadata_experimental")]
            rollup_cache: Arc::new(std::sync::RwLock::new(HashMap::new())),
            limit,
            current: Arc::new(AtomicUsize::new(0)),
            metadata_origin,
            data_origin,
        }
    }

    fn limit_bytes(&self) -> usize {
        self.limit * 1024 * 1024
    }

    /// Returns the current bytes stored in the LRU cache.
    pub fn used_bytes(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    /// Returns the configured LRU cache limit in bytes.
    pub fn limit_bytes_total(&self) -> usize {
        self.limit_bytes()
    }

    /// Evict LRU cache entries until usage is at or below
    /// `target_fraction` of the configured limit (e.g. 0.75 keeps 25%
    /// free).  Intended to be called post-commit so that subsequent
    /// layer loads find free space without needing inline eviction.
    ///
    /// This method uses `blocking_lock` and must NOT be called from
    /// within a tokio async context.
    pub fn evict_to_target(&self, target_fraction: f64) {
        let limit = self.limit_bytes();
        if limit == 0 {
            return;
        }
        let target = (limit as f64 * target_fraction) as usize;
        let current = self.current.load(Ordering::Relaxed);
        if current > target {
            let mut cache = self.cache.blocking_lock();
            let current = self.current.load(Ordering::Relaxed);
            if current > target {
                ensure_additional_cache_space(&mut cache, &self.current, current - target);
            }
        }
    }
}

impl<M: ArchiveMetadataBackend, D: ArchiveBackend> LruArchiveBackend<M, D> {
    async fn layer_fits_in_cache(&self, id: [u32; 5]) -> io::Result<bool> {
        let limit = self.limit_bytes();
        Ok(limit != 0 && self.layer_size(id).await? as usize <= limit)
    }
}

fn ensure_additional_cache_space(
    cache: &mut LruCache<[u32; 5], CacheEntry>,
    current: &AtomicUsize,
    mut required: usize,
) {
    if required == 0 {
        return;
    }

    loop {
        let peek = cache
            .peek_lru()
            .expect("cache is empty but stored entries were expected");
        if peek.1.is_resolving() {
            // this is a resolving entry, we don't want to pop it.
            let id = peek.0.clone();
            cache.promote(&id);
            continue;
        }
        // at this point the lru item is not resolving
        let entry = cache
            .pop_lru()
            .expect("cache is empty but stored entries were expected")
            .1;
        if let CacheEntry::Resolved(entry) = entry {
            current.fetch_sub(entry.len(), Ordering::Relaxed);
            if entry.len() >= required {
                // done!
                return;
            }

            // more needs to be popped
            required -= entry.len();
        } else {
            panic!("expected resolved entry but got a resolving");
        }
    }
}

fn ensure_enough_cache_space(
    cache: &mut LruCache<[u32; 5], CacheEntry>,
    limit: usize,
    current: &AtomicUsize,
    required: usize,
) -> bool {
    if required > limit {
        // this entry is too big for the cache
        return false;
    }

    let cur = current.load(Ordering::Relaxed);
    let remaining = limit.saturating_sub(cur);
    if remaining < required {
        // we need to clean up some cache space to fit this entry
        ensure_additional_cache_space(cache, current, required - remaining);
    }

    true
}

fn drop_from_cache(
    cache: &mut LruCache<[u32; 5], CacheEntry>,
    id: [u32; 5],
    current: &AtomicUsize,
) {
    if let Some(CacheEntry::Resolved(bytes)) = cache.pop(&id) {
        current.fetch_sub(bytes.len(), Ordering::Relaxed);
    }
}

#[async_trait]
impl<M: ArchiveMetadataBackend, D: ArchiveBackend> ArchiveBackend for LruArchiveBackend<M, D> {
    type Read = Either<BytesAsyncReader, D::Read>;
    async fn get_layer_bytes(&self, id: [u32; 5]) -> io::Result<Bytes> {
        let mut cache = self.cache.lock().await;
        let cached = cache.get(&id).cloned();

        match cached {
            Some(CacheEntry::Resolved(bytes)) => Ok(bytes),
            Some(CacheEntry::Resolving(barrier)) => {
                // someone is already looking up this layer. we'll wait for them to be done.
                std::mem::drop(cache);
                let guard = barrier.read().await;
                match guard.as_ref().unwrap() {
                    Ok(bytes) => Ok(bytes.clone()),
                    Err(kind) => Err(io::Error::new(*kind, "layer resolve failed")),
                }
            }
            None => {
                // nobody is looking this up yet, it is up to us.
                let barrier = Arc::new(tokio::sync::RwLock::new(None));
                let mut result = barrier.write().await;
                cache.get_or_insert(id, || CacheEntry::Resolving(barrier.clone()));

                // drop the cache while doing the lookup
                std::mem::drop(cache);
                let lookup = self.data_origin.get_layer_bytes(id).await;

                *result = Some(lookup.as_ref().map_err(|e| e.kind()).cloned());

                // reacquire cache
                let mut cache = self.cache.lock().await;
                match lookup {
                    Ok(bytes) => {
                        if ensure_enough_cache_space(
                            &mut *cache,
                            self.limit_bytes(),
                            &self.current,
                            bytes.len(),
                        ) {
                            let cached = cache
                                .get_mut(&id)
                                .expect("layer resolving entry not found in cache");
                            *cached = CacheEntry::Resolved(bytes.clone());
                            self.current.fetch_add(bytes.len(), Ordering::Relaxed);
                        } else {
                            // this entry is uncachable. Just remove the resolving entry
                            drop_from_cache(&mut *cache, id, &self.current);
                        }
                        Ok(bytes)
                    }
                    Err(e) => {
                        drop_from_cache(&mut *cache, id, &self.current);

                        Err(e)
                    }
                }
            }
        }
    }
    async fn get_layer_structure_bytes(
        &self,
        id: [u32; 5],
        file_type: LayerFileEnum,
    ) -> io::Result<Option<Bytes>> {
        if self.layer_fits_in_cache(id).await? {
            let bytes = self.get_layer_bytes(id).await?;
            let archive = Archive::parse(bytes);
            Ok(archive.slice_for(file_type))
        } else {
            self.data_origin
                .get_layer_structure_bytes(id, file_type)
                .await
        }
    }
    async fn layer_is_visible(&self, id: [u32; 5]) -> io::Result<bool> {
        self.data_origin.layer_is_visible(id).await
    }
    async fn store_layer_file(&self, id: [u32; 5], bytes: Bytes) -> io::Result<()> {
        loop {
            let mut cache = self.cache.lock().await;
            if let Some(CacheEntry::Resolving(barrier)) = cache.peek(&id).cloned() {
                // Let an in-flight old read finish before replacing this id; it
                // needs to reacquire the cache lock to publish its result.
                std::mem::drop(cache);
                let _ = barrier.read().await;
                continue;
            }
            if cache.contains(&id) {
                drop_from_cache(&mut *cache, id, &self.current);
            }

            // Hold the cache lock across the atomic backend replacement so no
            // reader can repopulate stale bytes in the eviction/write window.
            self.data_origin.store_layer_file(id, bytes.clone()).await?;
            let len = bytes.len();
            if ensure_enough_cache_space(&mut *cache, self.limit_bytes(), &self.current, len) {
                cache.put(id, CacheEntry::Resolved(bytes));
                self.current.fetch_add(len, Ordering::Relaxed);
            }
            return Ok(());
        }
    }
    async fn store_layer_file_if_absent(&self, id: [u32; 5], bytes: Bytes) -> io::Result<()> {
        loop {
            let mut cache = self.cache.lock().await;
            if let Some(CacheEntry::Resolving(barrier)) = cache.peek(&id).cloned() {
                std::mem::drop(cache);
                let _ = barrier.read().await;
                continue;
            }
            if cache.contains(&id) || self.metadata_origin.layer_exists(id).await? {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "archive layer already exists",
                ));
            }
            self.data_origin
                .store_layer_file_if_absent(id, bytes.clone())
                .await?;
            let len = bytes.len();
            if ensure_enough_cache_space(&mut *cache, self.limit_bytes(), &self.current, len) {
                cache.put(id, CacheEntry::Resolved(bytes));
                self.current.fetch_add(len, Ordering::Relaxed);
            }
            return Ok(());
        }
    }
    async fn replace_layer_file_if_matches(
        &self,
        id: [u32; 5],
        expected: Bytes,
        replacement: Bytes,
    ) -> io::Result<()> {
        loop {
            let mut cache = self.cache.lock().await;
            if let Some(CacheEntry::Resolving(barrier)) = cache.peek(&id).cloned() {
                std::mem::drop(cache);
                let _ = barrier.read().await;
                continue;
            }
            self.data_origin
                .replace_layer_file_if_matches(id, expected.clone(), replacement.clone())
                .await?;
            drop_from_cache(&mut cache, id, &self.current);
            let len = replacement.len();
            if ensure_enough_cache_space(&mut cache, self.limit_bytes(), &self.current, len) {
                cache.put(id, CacheEntry::Resolved(replacement));
                self.current.fetch_add(len, Ordering::Relaxed);
            }
            return Ok(());
        }
    }
    async fn publish_layer_files_atomically(
        &self,
        layers: Vec<([u32; 5], Bytes)>,
    ) -> io::Result<()> {
        loop {
            let mut cache = self.cache.lock().await;
            let resolving = layers
                .iter()
                .find_map(|(id, _)| match cache.peek(id).cloned() {
                    Some(CacheEntry::Resolving(barrier)) => Some(barrier),
                    _ => None,
                });
            if let Some(barrier) = resolving {
                std::mem::drop(cache);
                let _ = barrier.read().await;
                continue;
            }

            self.data_origin
                .publish_layer_files_atomically(layers.clone())
                .await?;
            for (id, bytes) in layers {
                drop_from_cache(&mut cache, id, &self.current);
                let len = bytes.len();
                if ensure_enough_cache_space(&mut cache, self.limit_bytes(), &self.current, len) {
                    cache.put(id, CacheEntry::Resolved(bytes));
                    self.current.fetch_add(len, Ordering::Relaxed);
                }
            }
            return Ok(());
        }
    }
    async fn remove_layer_file(&self, id: [u32; 5]) -> io::Result<()> {
        loop {
            let mut cache = self.cache.lock().await;
            if let Some(CacheEntry::Resolving(barrier)) = cache.peek(&id).cloned() {
                std::mem::drop(cache);
                let _ = barrier.read().await;
                continue;
            }
            if cache.contains(&id) {
                drop_from_cache(&mut *cache, id, &self.current);
            }
            return self.data_origin.remove_layer_file(id).await;
        }
    }
    async fn read_layer_structure_bytes_from(
        &self,
        id: [u32; 5],
        file_type: LayerFileEnum,
        read_from: usize,
    ) -> io::Result<Self::Read> {
        if self.layer_fits_in_cache(id).await? {
            let mut bytes = self
                .get_layer_structure_bytes(id, file_type)
                .await?
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::NotFound, "slice not found in archive")
                })?;
            bytes.advance(read_from);

            Ok(Either::Left(BytesAsyncReader(bytes)))
        } else {
            Ok(Either::Right(
                self.data_origin
                    .read_layer_structure_bytes_from(id, file_type, read_from)
                    .await?,
            ))
        }
    }
}

#[async_trait]
impl<M: ArchiveMetadataBackend, D: ArchiveBackend> ArchiveMetadataBackend
    for LruArchiveBackend<M, D>
{
    async fn get_layer_names(&self) -> io::Result<Vec<[u32; 5]>> {
        self.metadata_origin.get_layer_names().await
    }
    async fn layer_exists(&self, id: [u32; 5]) -> io::Result<bool> {
        if let Some(CacheEntry::Resolved(_)) = self.cache.lock().await.peek(&id) {
            Ok(true)
        } else {
            self.metadata_origin.layer_exists(id).await
        }
    }
    async fn layer_size(&self, id: [u32; 5]) -> io::Result<u64> {
        if let Some(CacheEntry::Resolved(bytes)) = self.cache.lock().await.peek(&id) {
            Ok(bytes.len() as u64)
        } else {
            self.metadata_origin.layer_size(id).await
        }
    }
    async fn layer_file_exists(&self, id: [u32; 5], file_type: LayerFileEnum) -> io::Result<bool> {
        if let Some(CacheEntry::Resolved(bytes)) = self.cache.lock().await.peek(&id) {
            let header = ArchiveFilePresenceHeader::new(bytes.clone().get_u64());
            Ok(header.is_present(file_type))
        } else {
            self.metadata_origin.layer_file_exists(id, file_type).await
        }
    }
    async fn get_layer_structure_size(
        &self,
        id: [u32; 5],
        file_type: LayerFileEnum,
    ) -> io::Result<usize> {
        if let Some(CacheEntry::Resolved(bytes)) = self.cache.lock().await.peek(&id) {
            let (header, _) = ArchiveHeader::parse(bytes.clone());

            if let Some(size) = header.size_of(file_type) {
                Ok(size)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "structure {file_type:?} not found in layer {}",
                        name_to_string(id)
                    ),
                ))
            }
        } else {
            self.metadata_origin
                .get_layer_structure_size(id, file_type)
                .await
        }
    }
    async fn get_rollup(&self, id: [u32; 5]) -> io::Result<Option<[u32; 5]>> {
        #[cfg(feature = "rollup_metadata_experimental")]
        {
            if let Some(cached) = self.rollup_cache.read().unwrap().get(&id) {
                return Ok(*cached);
            }
        }
        let result = self.metadata_origin.get_rollup(id).await?;
        #[cfg(feature = "rollup_metadata_experimental")]
        {
            self.rollup_cache.write().unwrap().insert(id, result);
        }
        Ok(result)
    }
    async fn set_rollup(&self, id: [u32; 5], rollup: [u32; 5]) -> io::Result<()> {
        self.metadata_origin.set_rollup(id, rollup).await?;
        #[cfg(feature = "rollup_metadata_experimental")]
        {
            self.rollup_cache.write().unwrap().insert(id, Some(rollup));
        }
        Ok(())
    }

    async fn get_rollup_record(&self, id: [u32; 5]) -> io::Result<Option<(u32, [u32; 5])>> {
        self.metadata_origin.get_rollup_record(id).await
    }

    async fn set_rollup_record(
        &self,
        id: [u32; 5],
        rollup: [u32; 5],
        version: u32,
    ) -> io::Result<()> {
        self.metadata_origin
            .set_rollup_record(id, rollup, version)
            .await?;
        #[cfg(feature = "rollup_metadata_experimental")]
        {
            self.rollup_cache.write().unwrap().insert(id, Some(rollup));
        }
        Ok(())
    }

    async fn get_parent(&self, id: [u32; 5]) -> io::Result<Option<[u32; 5]>> {
        if let Some(parent_bytes) = self
            .get_layer_structure_bytes(id, LayerFileEnum::Parent)
            .await?
        {
            let parent_string = std::str::from_utf8(&parent_bytes[..40]).unwrap();
            Ok(Some(string_to_name(parent_string).unwrap()))
        } else {
            Ok(None)
        }
    }
}

pub enum ConstructionFileState {
    UnderConstruction {
        bytes: BytesMut,
        writer_claimed: bool,
    },
    Finalizing,
    Finalized(Bytes),
}

pub struct ConstructionFile {
    state: Arc<RwLock<ConstructionFileState>>,
    writer: bool,
}

impl Clone for ConstructionFile {
    fn clone(&self) -> Self {
        // Cloning a file handle must never clone its exclusive write lease.
        Self {
            state: self.state.clone(),
            writer: false,
        }
    }
}

impl ConstructionFile {
    fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(ConstructionFileState::UnderConstruction {
                bytes: BytesMut::new(),
                writer_claimed: false,
            })),
            writer: false,
        }
    }

    fn new_finalized(bytes: Bytes) -> Self {
        Self {
            state: Arc::new(RwLock::new(ConstructionFileState::Finalized(bytes))),
            writer: false,
        }
    }

    fn is_finalized(&self) -> bool {
        let guard = self.state.read().unwrap();
        if let ConstructionFileState::Finalized(_) = &*guard {
            true
        } else {
            false
        }
    }

    fn finalized_buf(self) -> Bytes {
        let guard = self.state.read().unwrap();
        if let ConstructionFileState::Finalized(bytes) = &*guard {
            bytes.clone()
        } else {
            panic!("tried to get the finalized buf from an unfinalized ConstructionFile");
        }
    }
}

#[async_trait]
impl FileStore for ConstructionFile {
    type Write = Self;
    async fn open_write(&self) -> io::Result<Self::Write> {
        let mut guard = self.state.write().unwrap();
        match &mut *guard {
            ConstructionFileState::UnderConstruction { writer_claimed, .. } if !*writer_claimed => {
                *writer_claimed = true;
                Ok(Self {
                    state: self.state.clone(),
                    writer: true,
                })
            }
            _ => Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "archive construction file already has a writer",
            )),
        }
    }
}

impl AsyncWrite for ConstructionFile {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        if !self.writer {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "archive construction file handle has no write lease",
            )));
        }
        let mut guard = self.state.write().unwrap();
        match &mut *guard {
            ConstructionFileState::UnderConstruction { bytes, .. } => {
                bytes.put_slice(buf);

                Poll::Ready(Ok(buf.len()))
            }
            _ => Poll::Ready(Err(io::Error::new(
                io::ErrorKind::Other,
                "file already written",
            ))),
        }
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        // noop
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        // noop
        Poll::Ready(Ok(()))
    }
}

#[async_trait]
impl SyncableFile for ConstructionFile {
    async fn sync_all(self) -> io::Result<()> {
        if !self.writer {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "archive construction file handle has no write lease",
            ));
        }
        let mut guard = self.state.write().unwrap();
        let mut state = ConstructionFileState::Finalizing;
        std::mem::swap(&mut state, &mut *guard);

        match state {
            ConstructionFileState::UnderConstruction { bytes, .. } => {
                let buf = bytes.freeze();
                *guard = ConstructionFileState::Finalized(buf);

                Ok(())
            }
            _ => {
                *guard = state;
                Err(io::Error::new(io::ErrorKind::Other, "file already written"))
            }
        }
    }
}

impl AsyncRead for ConstructionFile {
    fn poll_read(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let mut guard = self.state.write().unwrap();
        match &mut *guard {
            ConstructionFileState::Finalized(x) => {
                let slice = if buf.remaining() > x.len() {
                    x.split_to(x.len())
                } else {
                    x.split_to(buf.remaining())
                };
                buf.put_slice(slice.as_ref());

                Poll::Ready(Ok(()))
            }
            _ => Poll::Ready(Err(io::Error::new(
                io::ErrorKind::Other,
                "file not yet written",
            ))),
        }
    }
}

#[async_trait]
impl FileLoad for ConstructionFile {
    type Read = Self;

    async fn exists(&self) -> io::Result<bool> {
        let guard = self.state.read().unwrap();
        Ok(matches!(&*guard, ConstructionFileState::Finalized(_)))
    }
    async fn size(&self) -> io::Result<usize> {
        let guard = self.state.read().unwrap();
        match &*guard {
            ConstructionFileState::Finalized(x) => Ok(x.len()),
            _ => Err(io::Error::new(
                io::ErrorKind::NotFound,
                "file not finalized",
            )),
        }
    }

    async fn open_read_from(&self, offset: usize) -> io::Result<Self::Read> {
        let guard = self.state.read().unwrap();
        match &*guard {
            ConstructionFileState::Finalized(data) => {
                let mut data = data.clone();
                if data.len() < offset {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "offset is beyond end of file",
                    ))
                } else {
                    data.advance(offset);
                    // this is suspicious, why would we need a lock here? Maybe we should have a different reader type from the file type
                    Ok(ConstructionFile {
                        state: Arc::new(RwLock::new(ConstructionFileState::Finalized(data))),
                        writer: false,
                    })
                }
            }
            _ => Err(io::Error::new(
                io::ErrorKind::NotFound,
                "file not finalized",
            )),
        }
    }

    async fn map(&self) -> io::Result<Bytes> {
        let guard = self.state.read().unwrap();
        match &*guard {
            ConstructionFileState::Finalized(x) => Ok(x.clone()),
            _ => Err(io::Error::new(
                io::ErrorKind::NotFound,
                "file not finalized",
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArchiveFilePresenceHeader {
    present_files: SmallBitArray,
}

impl ArchiveFilePresenceHeader {
    pub fn new(val: u64) -> Self {
        Self {
            present_files: SmallBitArray::new(val),
        }
    }

    pub fn from_present<I: Iterator<Item = LayerFileEnum>>(present_files: I) -> Self {
        let mut val = 0;

        for file in present_files {
            val |= 1 << (u64::BITS - file as u32 - 1);
        }

        Self::new(val)
    }

    pub fn is_present(&self, file: LayerFileEnum) -> bool {
        self.present_files.get(file as usize)
    }

    pub fn inner(&self) -> u64 {
        self.present_files.inner()
    }

    pub fn file_index(&self, file: LayerFileEnum) -> Option<usize> {
        if !self.is_present(file) {
            return None;
        }

        Some(self.present_files.rank1(file as usize) - 1)
    }
}

#[derive(Debug, Clone)]
pub struct ArchiveHeader {
    file_presence: ArchiveFilePresenceHeader,
    file_offsets: MonotonicLogArray,
}

impl ArchiveHeader {
    pub fn parse(mut bytes: Bytes) -> (Self, Bytes) {
        let file_presence = ArchiveFilePresenceHeader::new(bytes.get_u64());
        let (file_offsets, remainder) = MonotonicLogArray::parse_header_first(bytes)
            .expect("unable to parse structure offsets");

        (
            Self {
                file_presence,
                file_offsets,
            },
            remainder,
        )
    }

    pub async fn parse_from_reader<R: AsyncRead + Unpin>(reader: &mut R) -> io::Result<Self> {
        let file_presence = ArchiveFilePresenceHeader::new(reader.read_u64().await?);
        let mut logarray_bytes = BytesMut::new();
        logarray_bytes.resize(8, 0);
        reader.read_exact(&mut logarray_bytes[0..8]).await?;
        let len = logarray_length_from_control_word(&logarray_bytes[0..8]);
        logarray_bytes.reserve(len);
        unsafe {
            logarray_bytes.set_len(8 + len);
        }
        reader.read_exact(&mut logarray_bytes[8..]).await?;

        let (file_offsets, _) =
            MonotonicLogArray::parse_header_first(logarray_bytes.freeze()).expect("what the heck");

        Ok(Self {
            file_presence,
            file_offsets,
        })
    }

    pub fn range_for(&self, file: LayerFileEnum) -> Option<Range<usize>> {
        if let Some(file_index) = self.file_presence.file_index(file) {
            let start: usize = if file_index == 0 {
                0
            } else {
                self.file_offsets.entry(file_index - 1) as usize
            };

            let end: usize = self.file_offsets.entry(file_index) as usize;

            Some(start..end)
        } else {
            None
        }
    }

    pub fn size_of(&self, file: LayerFileEnum) -> Option<usize> {
        self.range_for(file).map(|range| range.end - range.start)
    }
}

pub struct Archive {
    pub header: ArchiveHeader,
    pub data: Bytes,
}

impl Archive {
    pub fn parse(bytes: Bytes) -> Self {
        let (header, data) = ArchiveHeader::parse(bytes);

        Self { header, data }
    }

    pub async fn parse_from_reader<R: AsyncRead + Unpin>(reader: &mut R) -> io::Result<Self> {
        let header = ArchiveHeader::parse_from_reader(reader).await?;
        let data_len = header.file_offsets.entry(header.file_offsets.len() - 1) as usize;
        let mut data = BytesMut::with_capacity(data_len);
        data.reserve(data_len);
        unsafe { data.set_len(data_len) };
        reader.read_exact(&mut data[..]).await?;

        Ok(Self {
            header,
            data: data.freeze(),
        })
    }

    pub fn slice_for(&self, file: LayerFileEnum) -> Option<Bytes> {
        self.header
            .range_for(file)
            .map(|range| self.data.slice(range))
    }

    pub fn size_of(&self, file: LayerFileEnum) -> Option<usize> {
        self.header.size_of(file)
    }
}

#[derive(Clone)]
pub struct PersistentFileSlice<M, D> {
    metadata_backend: M,
    data_backend: D,
    layer_id: [u32; 5],
    file_type: LayerFileEnum,
}

impl<M, D> PersistentFileSlice<M, D> {
    fn new(
        metadata_backend: M,
        data_backend: D,
        layer_id: [u32; 5],
        file_type: LayerFileEnum,
    ) -> Self {
        Self {
            metadata_backend,
            data_backend,
            layer_id,
            file_type,
        }
    }
}

pub struct ArchiveSliceReader {
    file: File,
    remaining: usize,
}

impl ArchiveSliceReader {
    pub fn new(file: File, remaining: usize) -> Self {
        Self { file, remaining }
    }

    pub fn end_early(&mut self, end: usize) {
        self.remaining -= end;
    }
}

impl AsyncRead for ArchiveSliceReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.remaining == 0 {
            return Poll::Ready(Ok(()));
        }

        let before_len = buf.filled().len();
        let read = AsyncRead::poll_read(Pin::new(&mut self.file), cx, buf);
        if let Poll::Pending = read {
            return Poll::Pending;
        }

        let after_len = buf.filled().len();
        let read_len = after_len - before_len;

        if read_len > self.remaining {
            buf.set_filled(before_len + self.remaining);
            self.remaining = 0;
        } else {
            self.remaining -= read_len;
        }

        Poll::Ready(Ok(()))
    }
}

#[async_trait]
impl<M: ArchiveMetadataBackend, D: ArchiveBackend> FileLoad for PersistentFileSlice<M, D> {
    type Read = D::Read;

    async fn exists(&self) -> io::Result<bool> {
        self.metadata_backend
            .layer_file_exists(self.layer_id, self.file_type)
            .await
    }

    async fn size(&self) -> io::Result<usize> {
        self.metadata_backend
            .get_layer_structure_size(self.layer_id, self.file_type)
            .await
    }

    async fn open_read_from(&self, offset: usize) -> io::Result<Self::Read> {
        self.data_backend
            .read_layer_structure_bytes_from(self.layer_id, self.file_type, offset)
            .await
    }

    async fn map(&self) -> io::Result<Bytes> {
        self.data_backend
            .get_layer_structure_bytes(self.layer_id, self.file_type)
            .await?
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "slice not found in archive"))
    }

    async fn map_if_exists(&self) -> io::Result<Option<Bytes>> {
        self.data_backend
            .get_layer_structure_bytes(self.layer_id, self.file_type)
            .await
    }
}

// This is some pretty ridiculous contrived logic but it saves having to refactor some other places which should just take a rollup id in the first place.
#[derive(Clone)]
pub struct ArchiveRollupFile<M> {
    layer_id: [u32; 5],
    metadata_backend: M,
}

#[async_trait]
impl<M: ArchiveMetadataBackend> FileLoad for ArchiveRollupFile<M> {
    type Read = BytesAsyncReader;

    async fn exists(&self) -> io::Result<bool> {
        Ok(self
            .metadata_backend
            .get_rollup(self.layer_id)
            .await?
            .is_some())
    }

    async fn size(&self) -> io::Result<usize> {
        if self
            .metadata_backend
            .get_rollup(self.layer_id)
            .await?
            .is_some()
        {
            Ok(self.map().await?.len())
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "layer has no rollup",
            ))
        }
    }

    async fn open_read_from(&self, offset: usize) -> io::Result<Self::Read> {
        let mut bytes = self.map().await?;
        bytes.advance(offset);
        Ok(BytesAsyncReader(bytes))
    }

    async fn map(&self) -> io::Result<Bytes> {
        let record = self.metadata_backend.get_rollup_record(self.layer_id).await?;
        if let Some((version, id)) = record {
            let mut bytes = Vec::with_capacity(42);
            bytes.extend_from_slice(format!("{version}\n").as_bytes());
            bytes.extend_from_slice(name_to_string(id).as_bytes());
            Ok(bytes.into())
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "layer has no rollup",
            ))
        }
    }
}

#[async_trait]
impl<M: ArchiveMetadataBackend + Unpin> FileStore for ArchiveRollupFile<M> {
    type Write = ArchiveRollupFileWriter<M>;
    async fn open_write(&self) -> io::Result<Self::Write> {
        Ok(ArchiveRollupFileWriter {
            layer_id: self.layer_id,
            data: BytesMut::new(),
            metadata_backend: self.metadata_backend.clone(),
        })
    }
}

pub struct ArchiveRollupFileWriter<M> {
    layer_id: [u32; 5],
    data: BytesMut,
    metadata_backend: M,
}

impl<M: ArchiveMetadataBackend + Unpin> AsyncWrite for ArchiveRollupFileWriter<M> {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        self.get_mut().data.extend_from_slice(buf);

        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        Poll::Ready(Ok(()))
    }
}

#[async_trait]
impl<M: ArchiveMetadataBackend + Unpin> SyncableFile for ArchiveRollupFileWriter<M> {
    async fn sync_all(self) -> io::Result<()> {
        let rollup_string =
            String::from_utf8(self.data.to_vec()).expect("rollup id was not a string");
        // The first line is the registration's version, the second the rollup.
        let mut lines = rollup_string.lines();
        let version = lines
            .next()
            .and_then(|line| line.parse::<u32>().ok())
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "rollup record has no version")
            })?;
        let line = lines.next().unwrap();
        let rollup_id = string_to_name(&line)?;

        self.metadata_backend
            .set_rollup_record(self.layer_id, rollup_id, version)
            .await
    }
}

#[derive(Clone)]
pub enum ArchiveLayerHandle<M, D> {
    Construction(ConstructionFile),
    Persistent(PersistentFileSlice<M, D>),
    Rollup(ArchiveRollupFile<M>),
}

#[async_trait]
impl<M: ArchiveMetadataBackend + Unpin, D: ArchiveBackend> FileStore for ArchiveLayerHandle<M, D> {
    type Write = ArchiveLayerHandleWriter<M>;
    async fn open_write(&self) -> io::Result<Self::Write> {
        Ok(match self {
            Self::Construction(c) => ArchiveLayerHandleWriter::Construction(c.open_write().await?),
            Self::Rollup(r) => ArchiveLayerHandleWriter::Rollup(r.open_write().await?),
            _ => panic!("cannot write to a persistent file slice"),
        })
    }
}

#[async_trait]
impl<M: ArchiveMetadataBackend, D: ArchiveBackend> FileLoad for ArchiveLayerHandle<M, D> {
    type Read = ArchiveLayerHandleReader<D::Read, BytesAsyncReader>;

    async fn exists(&self) -> io::Result<bool> {
        match self {
            Self::Construction(c) => c.exists().await,
            Self::Persistent(p) => p.exists().await,
            Self::Rollup(r) => r.exists().await,
        }
    }
    async fn size(&self) -> io::Result<usize> {
        match self {
            Self::Construction(c) => c.size().await,
            Self::Persistent(p) => p.size().await,
            Self::Rollup(r) => r.size().await,
        }
    }

    async fn open_read_from(&self, offset: usize) -> io::Result<Self::Read> {
        Ok(match self {
            Self::Construction(c) => {
                ArchiveLayerHandleReader::Construction(c.open_read_from(offset).await?)
            }
            Self::Persistent(p) => {
                ArchiveLayerHandleReader::Persistent(p.open_read_from(offset).await?)
            }
            Self::Rollup(r) => ArchiveLayerHandleReader::Rollup(r.open_read_from(offset).await?),
        })
    }

    async fn map(&self) -> io::Result<Bytes> {
        match self {
            Self::Construction(c) => c.map().await,
            Self::Persistent(p) => p.map().await,
            Self::Rollup(r) => r.map().await,
        }
    }
}

pub enum ArchiveLayerHandleReader<P, R> {
    Construction(ConstructionFile),
    Persistent(P),
    Rollup(R),
}

impl<P: AsyncRead + Unpin, R: AsyncRead + Unpin> AsyncRead for ArchiveLayerHandleReader<P, R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        match &mut *self {
            Self::Construction(c) => AsyncRead::poll_read(Pin::new(c), cx, buf),
            Self::Persistent(p) => AsyncRead::poll_read(Pin::new(p), cx, buf),
            Self::Rollup(r) => AsyncRead::poll_read(Pin::new(r), cx, buf),
        }
    }
}

pub enum ArchiveLayerHandleWriter<M> {
    Construction(ConstructionFile),
    Rollup(ArchiveRollupFileWriter<M>),
}

impl<M: ArchiveMetadataBackend + Unpin> AsyncWrite for ArchiveLayerHandleWriter<M> {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, io::Error>> {
        match &mut *self {
            Self::Construction(c) => AsyncWrite::poll_write(Pin::new(c), cx, buf),
            Self::Rollup(r) => AsyncWrite::poll_write(Pin::new(r), cx, buf),
        }
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        match &mut *self {
            Self::Construction(c) => AsyncWrite::poll_flush(Pin::new(c), cx),
            Self::Rollup(r) => AsyncWrite::poll_flush(Pin::new(r), cx),
        }
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<(), io::Error>> {
        match &mut *self {
            Self::Construction(c) => AsyncWrite::poll_shutdown(Pin::new(c), cx),
            Self::Rollup(r) => AsyncWrite::poll_shutdown(Pin::new(r), cx),
        }
    }
}

#[async_trait]
impl<M: ArchiveMetadataBackend + Unpin> SyncableFile for ArchiveLayerHandleWriter<M> {
    async fn sync_all(self) -> io::Result<()> {
        match self {
            Self::Construction(c) => c.sync_all().await,
            Self::Rollup(r) => r.sync_all().await,
        }
    }
}

type ArchiveLayerConstructionMap =
    Arc<RwLock<HashMap<[u32; 5], HashMap<LayerFileEnum, ConstructionFile>>>>;

#[derive(Clone)]
pub struct ArchiveLayerStore<M, D> {
    metadata_backend: M,
    data_backend: D,
    construction: ArchiveLayerConstructionMap,
}

impl<M, D> ArchiveLayerStore<M, D> {
    pub fn new(metadata_backend: M, data_backend: D) -> ArchiveLayerStore<M, D> {
        ArchiveLayerStore {
            metadata_backend,
            data_backend,
            construction: Default::default(),
        }
    }

    #[doc(hidden)]
    pub fn write_bytes(&self, name: [u32; 5], file: LayerFileEnum, bytes: Bytes) {
        let mut guard = self.construction.write().unwrap();
        if let Some(map) = guard.get_mut(&name) {
            if map.contains_key(&file) {
                panic!("tried to write bytes to an archive, but file is already open");
            }

            map.insert(file, ConstructionFile::new_finalized(bytes));
        } else {
            panic!("tried to write bytes to an archive, but layer is not under construction");
        }
    }
}

fn encode_archive_layer(files: HashMap<LayerFileEnum, Bytes>) -> io::Result<Bytes> {
    if files.contains_key(&LayerFileEnum::Rollup) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rollup metadata cannot be embedded in a layer archive",
        ));
    }

    let mut files: Vec<_> = files.into_iter().collect();
    files.sort_by_key(|(file_type, _)| *file_type);
    let presence_header =
        ArchiveFilePresenceHeader::from_present(files.iter().map(|(file_type, _)| *file_type));

    let mut offsets = LateLogArrayBufBuilder::new(BytesMut::new());
    let mut tally = 0usize;
    for (_, data) in &files {
        tally = tally.checked_add(data.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "layer archive size overflow")
        })?;
        offsets.push(u64::try_from(tally).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "layer archive is too large")
        })?);
    }

    let offsets_buf = offsets.finalize_header_first();
    let capacity = tally
        .checked_add(8)
        .and_then(|value| value.checked_add(offsets_buf.len()))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "archive size overflow"))?;
    let mut data_buf = BytesMut::with_capacity(capacity);
    data_buf.put_u64(presence_header.inner());
    data_buf.extend(offsets_buf);
    for (_, data) in files {
        data_buf.extend(data);
    }
    Ok(data_buf.freeze())
}

#[async_trait]
impl<M: ArchiveMetadataBackend + Unpin + 'static, D: ArchiveBackend + 'static> PersistentLayerStore
    for ArchiveLayerStore<M, D>
{
    type File = ArchiveLayerHandle<M, D>;

    async fn directories(&self) -> io::Result<Vec<[u32; 5]>> {
        let mut result = Vec::new();
        for name in self.metadata_backend.get_layer_names().await? {
            if self.data_backend.layer_is_visible(name).await? {
                result.push(name);
            }
        }

        {
            let guard = self.construction.read().unwrap();

            for name in guard.keys() {
                result.push(*name);
            }
        }

        result.sort();
        result.dedup();

        Ok(result)
    }

    async fn create_named_directory(&self, name: [u32; 5]) -> io::Result<[u32; 5]> {
        if self.data_backend.layer_is_visible(name).await?
            || self.metadata_backend.layer_exists(name).await?
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "archive layer already exists",
            ));
        }
        let mut guard = self.construction.write().unwrap();
        if guard.contains_key(&name) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "archive layer is already under construction",
            ));
        }
        guard.insert(name, HashMap::new());
        Ok(name)
    }

    async fn directory_exists(&self, name: [u32; 5]) -> io::Result<bool> {
        {
            let guard = self.construction.read().unwrap();
            if guard.contains_key(&name) {
                return Ok(true);
            }
        }

        Ok(self.data_backend.layer_is_visible(name).await?
            && self.metadata_backend.layer_exists(name).await?)
    }

    async fn get_file(&self, directory: [u32; 5], name: &str) -> io::Result<Self::File> {
        let file_type = FILENAME_ENUM_MAP.get(name).copied().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "unknown archive layer structure",
            )
        })?;
        if file_type == LayerFileEnum::Rollup {
            if !self.data_backend.layer_is_visible(directory).await? {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "archive layer not found",
                ));
            }
            // special case! This is always coming from disk, in its own file
            return Ok(ArchiveLayerHandle::Rollup(ArchiveRollupFile {
                layer_id: directory,
                metadata_backend: self.metadata_backend.clone(),
            }));
        }

        let construction_file = {
            let mut guard = self.construction.write().unwrap();
            guard.get_mut(&directory).map(|map| {
                map.entry(file_type)
                    .or_insert_with(ConstructionFile::new)
                    .clone()
            })
        };
        if let Some(file) = construction_file {
            return Ok(ArchiveLayerHandle::Construction(file));
        }
        if !self.data_backend.layer_is_visible(directory).await? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "archive layer not found",
            ));
        }
        Ok(ArchiveLayerHandle::Persistent(PersistentFileSlice::new(
            self.metadata_backend.clone(),
            self.data_backend.clone(),
            directory,
            file_type,
        )))
    }

    async fn file_exists(&self, directory: [u32; 5], file: &str) -> io::Result<bool> {
        let file_type = FILENAME_ENUM_MAP.get(file).copied().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "unknown archive layer structure",
            )
        })?;
        if file_type == LayerFileEnum::Rollup {
            // special case! This is always coming out of the persistent metadata
            return Ok(self.metadata_backend.get_rollup(directory).await?.is_some());
        }

        {
            let guard = self.construction.read().unwrap();
            if let Some(map) = guard.get(&directory) {
                return Ok(map.contains_key(&file_type));
            }
        }

        if !self.data_backend.layer_is_visible(directory).await? {
            return Ok(false);
        }
        self.metadata_backend
            .layer_file_exists(directory, file_type)
            .await
    }

    async fn abort_directory(&self, directory: [u32; 5]) -> io::Result<()> {
        let removed = self.construction.write().unwrap().remove(&directory);
        if removed.is_some() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot abort an exposed archive layer",
            ))
        }
    }

    async fn publish_imported_layers_shared(
        &self,
        layers: Vec<([u32; 5], HashMap<String, Bytes>)>,
    ) -> io::Result<()> {
        let mut encoded = Vec::with_capacity(layers.len());
        let mut ids = std::collections::HashSet::new();
        for (id, files) in layers {
            if !ids.insert(id) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "duplicate imported layer id",
                ));
            }
            if self.construction.read().unwrap().contains_key(&id)
                || self.data_backend.layer_is_visible(id).await?
                || self.metadata_backend.layer_exists(id).await?
            {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "refusing to publish over an existing archive layer",
                ));
            }
            let mut structures = HashMap::with_capacity(files.len());
            for (name, bytes) in files {
                let file_type = FILENAME_ENUM_MAP
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "unknown imported layer structure",
                        )
                    })?;
                if file_type == LayerFileEnum::Rollup {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "rollup metadata is not portable in a layer pack",
                    ));
                }
                if structures.insert(file_type, bytes).is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "duplicate imported layer structure",
                    ));
                }
            }
            encoded.push((id, encode_archive_layer(structures)?));
        }
        self.data_backend
            .publish_layer_files_atomically(encoded)
            .await
    }

    async fn write_proof_state_bytes(&self, directory: [u32; 5], bytes: &[u8]) -> io::Result<()> {
        let under_construction = self.construction.read().unwrap().contains_key(&directory);
        if under_construction {
            let file = self.get_file(directory, FILENAMES.proof_state).await?;
            let mut writer = file.open_write().await?;
            writer.write_all(bytes).await?;
            writer.flush().await?;
            return writer.sync_all().await;
        }
        if !self.data_backend.layer_is_visible(directory).await?
            || !self.metadata_backend.layer_exists(directory).await?
        {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "archive layer not found",
            ));
        }
        let expected = self.data_backend.get_layer_bytes(directory).await?;
        let archive = Archive::parse(expected.clone());
        if archive.slice_for(LayerFileEnum::ProofState).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "proof state already exists",
            ));
        }
        let mut structures = HashMap::new();
        for raw in 0..=LayerFileEnum::ProofState as u64 {
            let kind = LayerFileEnum::from_u64(raw).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "unknown archive file bit")
            })?;
            if kind == LayerFileEnum::Rollup || kind == LayerFileEnum::ProofState {
                continue;
            }
            if let Some(contents) = archive.slice_for(kind) {
                structures.insert(kind, contents);
            }
        }
        structures.insert(LayerFileEnum::ProofState, Bytes::copy_from_slice(bytes));
        let replacement = encode_archive_layer(structures)?;
        self.data_backend
            .replace_layer_file_if_matches(directory, expected, replacement)
            .await
    }

    async fn replace_proof_state_bytes_if_matches(
        &self,
        directory: [u32; 5],
        expected_proof: &[u8],
        replacement_proof: &[u8],
    ) -> io::Result<()> {
        if self.construction.read().unwrap().contains_key(&directory) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "cannot migrate an archive layer under construction",
            ));
        }
        let expected_archive = self.data_backend.get_layer_bytes(directory).await?;
        let archive = Archive::parse(expected_archive.clone());
        let current = archive
            .slice_for(LayerFileEnum::ProofState)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "proof state not found"))?;
        if current.as_ref() != expected_proof {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "proof state changed during migration",
            ));
        }
        let mut structures = HashMap::new();
        for raw in 0..=LayerFileEnum::ProofState as u64 {
            let kind = LayerFileEnum::from_u64(raw).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "unknown archive file bit")
            })?;
            if kind == LayerFileEnum::Rollup || kind == LayerFileEnum::ProofState {
                continue;
            }
            if let Some(contents) = archive.slice_for(kind) {
                structures.insert(kind, contents);
            }
        }
        structures.insert(
            LayerFileEnum::ProofState,
            Bytes::copy_from_slice(replacement_proof),
        );
        let replacement_archive = encode_archive_layer(structures)?;
        self.data_backend
            .replace_layer_file_if_matches(directory, expected_archive, replacement_archive)
            .await
    }

    async fn finalize(&self, directory: [u32; 5]) -> io::Result<()> {
        let files = {
            let mut guard = self.construction.write().unwrap();
            guard
                .remove(&directory)
                .expect("layer to be finalized was not found in construction map")
        };

        let files: HashMap<_, _> = files
            .into_iter()
            .filter(|(_file_type, file)| file.is_finalized())
            .map(|(file_type, file)| (file_type, file.finalized_buf()))
            .collect();
        let data_buf = encode_archive_layer(files)?;
        self.data_backend
            .store_layer_file_if_absent(directory, data_buf)
            .await
    }

    async fn layer_parent(&self, name: [u32; 5]) -> io::Result<Option<[u32; 5]>> {
        if !self.data_backend.layer_is_visible(name).await? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "archive layer not found",
            ));
        }
        self.metadata_backend.get_parent(name).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use tempfile::tempdir;

    const PROCESS_ROOT: &str = "TERMINUS_STORE_ARCHIVE_PROCESS_ROOT";
    const PROCESS_ID: &str = "TERMINUS_STORE_ARCHIVE_PROCESS_ID";
    const PROCESS_IDS: &str = "TERMINUS_STORE_ARCHIVE_PROCESS_IDS";
    const PROCESS_PAYLOAD: &str = "TERMINUS_STORE_ARCHIVE_PROCESS_PAYLOAD";

    fn test_id(n: u32) -> [u32; 5] {
        [
            n,
            n.wrapping_mul(3),
            n.wrapping_mul(5),
            n.wrapping_mul(7),
            n.wrapping_mul(11),
        ]
    }

    fn test_archive(payload: &[u8]) -> Bytes {
        let mut files = HashMap::new();
        files.insert(LayerFileEnum::PosSubjects, Bytes::copy_from_slice(payload));
        encode_archive_layer(files).unwrap()
    }

    #[tokio::test]
    async fn archive_publication_process_child() {
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
        let payload = std::env::var(PROCESS_PAYLOAD).unwrap();
        let backend = DirectoryArchiveBackend::new(root.into());
        let multiple = ids.len() > 1;
        let layers = ids
            .into_iter()
            .enumerate()
            .map(|(index, id)| {
                let payload = if multiple {
                    format!("{payload}{index}")
                } else {
                    payload.clone()
                };
                (id, test_archive(payload.as_bytes()))
            })
            .collect();
        let result = backend.publish_layer_files_atomically(layers).await;
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
    async fn archive_recovery_process_child() {
        let Ok(root) = std::env::var(PROCESS_ROOT) else {
            return;
        };
        let backend = DirectoryArchiveBackend::new(root.into());
        let _ = backend.get_layer_names().await;
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
            .arg("storage::archive::tests::archive_publication_process_child")
            .arg("--nocapture")
            .env(PROCESS_ROOT, root)
            .env(PROCESS_ID, name_to_string(id))
            .env(PROCESS_PAYLOAD, payload)
            .env("RUST_TEST_THREADS", "1")
            .env_remove("TERMINUS_STORE_ARCHIVE_CRASH_POINT")
            .env_remove("TERMINUS_STORE_ARCHIVE_FAIL_POST_COMMIT_FSYNC")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(point) = crash {
            command.env("TERMINUS_STORE_ARCHIVE_CRASH_POINT", point);
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
            .arg("storage::archive::tests::archive_recovery_process_child")
            .arg("--nocapture")
            .env(PROCESS_ROOT, root)
            .env("RUST_TEST_THREADS", "1")
            .env_remove("TERMINUS_STORE_ARCHIVE_CRASH_POINT")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(point) = crash {
            command.env("TERMINUS_STORE_ARCHIVE_CRASH_POINT", point);
        }
        command
    }

    #[tokio::test]
    async fn two_process_archive_publication_has_one_winner() {
        let root = tempdir().unwrap();
        let id = test_id(901);
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
        let backend = DirectoryArchiveBackend::new(root.path().to_path_buf());
        assert!(backend.layer_is_visible(id).await.unwrap());
        let bytes = backend.get_layer_bytes(id).await.unwrap();
        assert!(bytes == test_archive(b"first") || bytes == test_archive(b"second"));
    }

    #[tokio::test]
    async fn archive_crash_points_recover_to_atomic_visibility() {
        for (point, committed) in [
            ("after_staging_fsync", false),
            ("after_visibility_marker_rename", false),
            ("after_visibility_marker_fsync", false),
            ("after_target_rename", false),
            ("after_commit_marker_rename", true),
        ] {
            let root = tempdir().unwrap();
            let id = test_id(950 + point.len() as u32);
            let output = publication_child(root.path(), id, "payload", Some(point))
                .output()
                .unwrap();
            assert!(
                !output.status.success(),
                "crash hook {} did not terminate child",
                point
            );
            let backend = DirectoryArchiveBackend::new(root.path().to_path_buf());
            let layers = backend.get_layer_names().await.unwrap();
            assert_eq!(layers.contains(&id), committed, "crash point {point}");
            assert_eq!(backend.layer_is_visible(id).await.unwrap(), committed);
        }
    }

    #[tokio::test]
    async fn archive_multi_layer_process_crashes_are_all_or_none_to_readers() {
        let ids = [test_id(1001), test_id(1002), test_id(1003)];
        for (point, committed) in [
            ("after_staging_fsync", false),
            ("after_visibility_marker_fsync", false),
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
            let backend = DirectoryArchiveBackend::new(root.path().to_path_buf());
            let visible_before_recovery =
                futures::future::try_join_all(ids.iter().map(|id| backend.layer_is_visible(*id)))
                    .await
                    .unwrap();
            let visible_count = visible_before_recovery
                .iter()
                .filter(|visible| **visible)
                .count();
            assert_eq!(visible_count, if committed { ids.len() } else { 0 });
            let listed = backend.get_layer_names().await.unwrap();
            assert_eq!(
                ids.iter().filter(|id| listed.contains(id)).count(),
                if committed { ids.len() } else { 0 }
            );
        }
    }

    #[tokio::test]
    async fn archive_overlapping_batches_have_one_cross_process_winner() {
        let root = tempdir().unwrap();
        let first_ids = [test_id(1011), test_id(1012)];
        let second_ids = [test_id(1012), test_id(1013)];
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
        let backend = DirectoryArchiveBackend::new(root.path().to_path_buf());
        let listed = backend.get_layer_names().await.unwrap();
        assert!(listed.contains(&first_ids[1]));
        assert_eq!(
            listed.contains(&first_ids[0]),
            !listed.contains(&second_ids[1])
        );
        assert_eq!(listed.len(), 2);
    }

    #[tokio::test]
    async fn archive_crash_during_quarantine_cleans_hidden_and_marker_temps() {
        let root = tempdir().unwrap();
        let ids = [test_id(1021), test_id(1022)];
        let publication =
            batch_publication_child(root.path(), &ids, "batch", Some("after_target_rename"))
                .output()
                .unwrap();
        assert!(!publication.status.success());
        let recovery = recovery_child(root.path(), Some("during_quarantine"))
            .output()
            .unwrap();
        assert!(!recovery.status.success());

        let backend = DirectoryArchiveBackend::new(root.path().to_path_buf());
        assert!(backend.get_layer_names().await.unwrap().is_empty());
        for prefix in std::fs::read_dir(root.path()).unwrap() {
            let prefix = prefix.unwrap();
            if !prefix.file_type().unwrap().is_dir() {
                continue;
            }
            for entry in std::fs::read_dir(prefix.path()).unwrap() {
                let name = entry.unwrap().file_name().to_string_lossy().into_owned();
                assert!(
                    !name.contains(".recovery-")
                        && !name.contains(".import-")
                        && !name.contains(".marker-")
                );
            }
        }
    }

    #[tokio::test]
    async fn archive_post_commit_fsync_failure_is_explicitly_indeterminate() {
        let root = tempdir().unwrap();
        let id = test_id(999);
        let output = publication_child(root.path(), id, "payload", None)
            .env("TERMINUS_STORE_ARCHIVE_FAIL_POST_COMMIT_FSYNC", "1")
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("PROCESS_RESULT=indeterminate"));
        let backend = DirectoryArchiveBackend::new(root.path().to_path_buf());
        assert!(backend.layer_is_visible(id).await.unwrap());
    }

    #[tokio::test]
    async fn construction_file_allows_exactly_one_writer() {
        let file = ConstructionFile::new();
        let mut writer = file.open_write().await.unwrap();
        let error = match file.open_write().await {
            Ok(_) => panic!("second construction writer unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        writer.write_all(b"complete").await.unwrap();
        writer.sync_all().await.unwrap();
        assert_eq!(file.map().await.unwrap().as_ref(), b"complete");
    }

    #[test]
    fn parse_and_use_header() {
        let files_present = vec![
            LayerFileEnum::PredicateDictionaryBlocks,
            LayerFileEnum::NegObjects,
            LayerFileEnum::NodeValueIdMapBits,
            LayerFileEnum::Parent,
            LayerFileEnum::Rollup,
            LayerFileEnum::NodeDictionaryBlocks,
        ];

        let header = ArchiveFilePresenceHeader::from_present(files_present.iter().cloned());

        for file in files_present {
            assert!(header.is_present(file));
        }

        assert!(!header.is_present(LayerFileEnum::NodeDictionaryOffsets));
    }

    #[tokio::test]
    async fn archive_store_if_absent_never_replaces_existing_bytes() {
        let directory = tempdir().unwrap();
        let backend = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        let id = test_id(1);
        let original = test_archive(b"original");
        backend
            .store_layer_file_if_absent(id, original.clone())
            .await
            .unwrap();

        let error = backend
            .store_layer_file_if_absent(id, test_archive(b"replacement"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(backend.get_layer_bytes(id).await.unwrap(), original);
    }

    async fn race_proof_publication(first: &'static [u8], second: &'static [u8]) -> Bytes {
        let directory = tempdir().unwrap();
        let backend = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        let id = test_id(2);
        backend
            .store_layer_file_if_absent(id, test_archive(b"layer"))
            .await
            .unwrap();
        let store = ArchiveLayerStore::new(backend.clone(), backend.clone());

        let (left, right) = tokio::join!(
            store.write_proof_state_bytes(id, first),
            store.write_proof_state_bytes(id, second)
        );
        assert_eq!(usize::from(left.is_ok()) + usize::from(right.is_ok()), 1);
        let failure = if let Err(error) = left {
            error
        } else {
            right.unwrap_err()
        };
        assert_eq!(failure.kind(), io::ErrorKind::AlreadyExists);

        Archive::parse(backend.get_layer_bytes(id).await.unwrap())
            .slice_for(LayerFileEnum::ProofState)
            .unwrap()
    }

    #[tokio::test]
    async fn concurrent_same_proof_publication_has_one_winner() {
        assert_eq!(
            race_proof_publication(b"proof", b"proof").await,
            b"proof"[..]
        );
    }

    #[tokio::test]
    async fn concurrent_different_proof_publication_has_one_winner() {
        let proof = race_proof_publication(b"first", b"second").await;
        assert!(proof == b"first"[..] || proof == b"second"[..]);
    }

    #[tokio::test]
    async fn failed_archive_batch_publication_exposes_none_of_the_new_layers() {
        let directory = tempdir().unwrap();
        let backend = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        let first = test_id(3);
        let collision = test_id(4);
        let existing = test_archive(b"existing");
        backend
            .store_layer_file_if_absent(collision, existing.clone())
            .await
            .unwrap();

        let error = backend
            .publish_layer_files_atomically(vec![
                (first, test_archive(b"new")),
                (collision, test_archive(b"overwrite")),
            ])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert!(!backend.layer_is_visible(first).await.unwrap());
        assert_eq!(backend.get_layer_bytes(collision).await.unwrap(), existing);
        assert_eq!(backend.get_layer_names().await.unwrap(), vec![collision]);
    }

    #[tokio::test]
    async fn successful_archive_batch_becomes_visible_as_one_committed_set() {
        let directory = tempdir().unwrap();
        let backend = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        let first = test_id(5);
        let second = test_id(6);
        backend
            .publish_layer_files_atomically(vec![
                (first, test_archive(b"first")),
                (second, test_archive(b"second")),
            ])
            .await
            .unwrap();

        assert!(backend.layer_is_visible(first).await.unwrap());
        assert!(backend.layer_is_visible(second).await.unwrap());
        let mut names = backend.get_layer_names().await.unwrap();
        names.sort();
        assert_eq!(names, vec![first, second]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn overlapping_archive_batches_have_one_atomic_winner() {
        let directory = tempdir().unwrap();
        let first_backend = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        let second_backend = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        let first = test_id(9);
        let overlap = test_id(10);
        let second = test_id(11);

        let (left, right) = tokio::join!(
            first_backend.publish_layer_files_atomically(vec![
                (first, test_archive(b"first")),
                (overlap, test_archive(b"left-overlap")),
            ]),
            second_backend.publish_layer_files_atomically(vec![
                (overlap, test_archive(b"right-overlap")),
                (second, test_archive(b"second")),
            ])
        );
        assert_ne!(left.is_ok(), right.is_ok());
        let first_visible = first_backend.layer_is_visible(first).await.unwrap();
        let overlap_visible = first_backend.layer_is_visible(overlap).await.unwrap();
        let second_visible = first_backend.layer_is_visible(second).await.unwrap();
        assert!(overlap_visible);
        assert!(
            (first_visible && !second_visible) || (!first_visible && second_visible),
            "readers must observe exactly one complete overlapping transaction"
        );
    }

    #[tokio::test]
    async fn reopen_recovers_uncommitted_archive_target_and_preserves_commit() {
        let directory = tempdir().unwrap();
        let backend = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        let abandoned = test_id(12);
        let token = name_to_string(test_id(13));
        let target = backend.path_for_layer(abandoned);
        fs::create_dir_all(target.parent().unwrap()).await.unwrap();
        fs::write(&target, test_archive(b"abandoned"))
            .await
            .unwrap();
        fs::write(backend.path_for_import_marker(abandoned), token)
            .await
            .unwrap();
        assert!(!backend.layer_is_visible(abandoned).await.unwrap());

        let reopened = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        assert!(!reopened
            .get_layer_names()
            .await
            .unwrap()
            .contains(&abandoned));
        assert!(matches!(
            fs::metadata(&target).await,
            Err(error) if error.kind() == io::ErrorKind::NotFound
        ));

        let committed = test_id(14);
        reopened
            .publish_layer_files_atomically(vec![(committed, test_archive(b"committed"))])
            .await
            .unwrap();
        let reopened_again = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        assert!(reopened_again.layer_is_visible(committed).await.unwrap());
        assert!(reopened_again
            .get_layer_names()
            .await
            .unwrap()
            .contains(&committed));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn archive_recovery_and_normal_write_are_serialized_by_layer_lock() {
        let directory = tempdir().unwrap();
        let backend = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        let id = test_id(77);
        let token = name_to_string(test_id(78));
        let target = backend.path_for_layer(id);
        fs::create_dir_all(target.parent().unwrap()).await.unwrap();
        fs::write(&target, test_archive(b"abandoned"))
            .await
            .unwrap();
        fs::write(backend.path_for_import_marker(id), token)
            .await
            .unwrap();

        let recovery = backend.clone();
        let writer = backend.clone();
        let replacement = test_archive(b"replacement");
        let (recovered, written) = tokio::join!(
            recovery.get_layer_names(),
            writer.store_layer_file_if_absent(id, replacement.clone())
        );
        recovered.unwrap();
        if written.is_ok() {
            assert!(backend.layer_is_visible(id).await.unwrap());
            assert_eq!(backend.get_layer_bytes(id).await.unwrap(), replacement);
        } else {
            assert!(!backend.layer_is_visible(id).await.unwrap());
        }
    }

    #[tokio::test]
    async fn lru_conditional_replacement_invalidates_exact_entry_and_accounts_bytes() {
        let directory = tempdir().unwrap();
        let origin = DirectoryArchiveBackend::new(directory.path().to_path_buf());
        let first = test_id(7);
        let second = test_id(8);
        let old_first = Bytes::from_static(b"0123456789");
        let old_second = Bytes::from_static(b"abcdefghijklmnopqrst");
        origin
            .store_layer_file_if_absent(first, old_first.clone())
            .await
            .unwrap();
        origin
            .store_layer_file_if_absent(second, old_second.clone())
            .await
            .unwrap();
        let cached = LruArchiveBackend::new(origin.clone(), origin, 1);

        cached.get_layer_bytes(first).await.unwrap();
        cached.get_layer_bytes(second).await.unwrap();
        cached.get_layer_bytes(first).await.unwrap();
        assert_eq!(cached.used_bytes(), old_first.len() + old_second.len());

        let replacement = Bytes::from_static(b"replacement-123");
        cached
            .replace_layer_file_if_matches(first, old_first, replacement.clone())
            .await
            .unwrap();
        assert_eq!(cached.get_layer_bytes(first).await.unwrap(), replacement);
        assert_eq!(
            cached.used_bytes(),
            replacement.len() + old_second.len(),
            "replacement must evict and subtract the addressed entry, not the LRU entry"
        );
    }

    #[cfg(feature = "rollup_metadata_experimental")]
    mod rollup_metadata_cache_tests {
        use super::*;
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;

        /// Mock metadata backend that tracks get_rollup call counts.
        #[derive(Clone)]
        struct MockMetadataBackend {
            rollup_map: Arc<std::sync::RwLock<HashMap<[u32; 5], [u32; 5]>>>,
            get_rollup_calls: Arc<AtomicUsize>,
        }

        impl MockMetadataBackend {
            fn new() -> Self {
                Self {
                    rollup_map: Arc::new(std::sync::RwLock::new(HashMap::new())),
                    get_rollup_calls: Arc::new(AtomicUsize::new(0)),
                }
            }

            fn get_rollup_call_count(&self) -> usize {
                self.get_rollup_calls.load(Ordering::Relaxed)
            }
        }

        #[async_trait]
        impl ArchiveMetadataBackend for MockMetadataBackend {
            async fn get_layer_names(&self) -> io::Result<Vec<[u32; 5]>> {
                Ok(vec![])
            }
            async fn layer_exists(&self, _id: [u32; 5]) -> io::Result<bool> {
                Ok(true)
            }
            async fn layer_size(&self, _id: [u32; 5]) -> io::Result<u64> {
                Ok(100)
            }
            async fn layer_file_exists(
                &self,
                _id: [u32; 5],
                _file_type: LayerFileEnum,
            ) -> io::Result<bool> {
                Ok(false)
            }
            async fn get_layer_structure_size(
                &self,
                _id: [u32; 5],
                _file_type: LayerFileEnum,
            ) -> io::Result<usize> {
                Ok(0)
            }
            async fn get_rollup(&self, id: [u32; 5]) -> io::Result<Option<[u32; 5]>> {
                self.get_rollup_calls.fetch_add(1, Ordering::Relaxed);
                Ok(self.rollup_map.read().unwrap().get(&id).copied())
            }
            async fn set_rollup(&self, id: [u32; 5], rollup: [u32; 5]) -> io::Result<()> {
                self.rollup_map.write().unwrap().insert(id, rollup);
                Ok(())
            }
            async fn get_parent(&self, _id: [u32; 5]) -> io::Result<Option<[u32; 5]>> {
                Ok(None)
            }
        }

        /// Minimal mock data backend (not used by rollup metadata tests).
        #[derive(Clone)]
        struct MockDataBackend;

        #[async_trait]
        impl ArchiveBackend for MockDataBackend {
            type Read = BytesAsyncReader;
            async fn get_layer_bytes(&self, _id: [u32; 5]) -> io::Result<Bytes> {
                Err(io::Error::new(io::ErrorKind::NotFound, "mock"))
            }
            async fn get_layer_structure_bytes(
                &self,
                _id: [u32; 5],
                _file_type: LayerFileEnum,
            ) -> io::Result<Option<Bytes>> {
                Ok(None)
            }
            async fn layer_is_visible(&self, _id: [u32; 5]) -> io::Result<bool> {
                Ok(false)
            }
            async fn store_layer_file(&self, _id: [u32; 5], _bytes: Bytes) -> io::Result<()> {
                Ok(())
            }
            async fn store_layer_file_if_absent(
                &self,
                _id: [u32; 5],
                _bytes: Bytes,
            ) -> io::Result<()> {
                Err(io::Error::new(io::ErrorKind::Unsupported, "mock"))
            }
            async fn replace_layer_file_if_matches(
                &self,
                _id: [u32; 5],
                _expected: Bytes,
                _replacement: Bytes,
            ) -> io::Result<()> {
                Err(io::Error::new(io::ErrorKind::Unsupported, "mock"))
            }
            async fn publish_layer_files_atomically(
                &self,
                _layers: Vec<([u32; 5], Bytes)>,
            ) -> io::Result<()> {
                Err(io::Error::new(io::ErrorKind::Unsupported, "mock"))
            }
            async fn remove_layer_file(&self, _id: [u32; 5]) -> io::Result<()> {
                Ok(())
            }
            async fn read_layer_structure_bytes_from(
                &self,
                _id: [u32; 5],
                _file_type: LayerFileEnum,
                _read_from: usize,
            ) -> io::Result<Self::Read> {
                Err(io::Error::new(io::ErrorKind::NotFound, "mock"))
            }
        }

        fn test_layer_id(n: u32) -> [u32; 5] {
            [n, 0, 0, 0, 0]
        }

        fn make_lru_backend(
            mock: MockMetadataBackend,
        ) -> LruArchiveBackend<MockMetadataBackend, MockDataBackend> {
            LruArchiveBackend::new(mock, MockDataBackend, 512)
        }

        #[tokio::test]
        async fn get_rollup_caches_result_on_second_call() {
            let mock = MockMetadataBackend::new();
            let layer_id = test_layer_id(1);
            let rollup_id = test_layer_id(99);
            mock.rollup_map.write().unwrap().insert(layer_id, rollup_id);

            let backend = make_lru_backend(mock.clone());

            let result1 = backend.get_rollup(layer_id).await.unwrap();
            assert_eq!(result1, Some(rollup_id));
            assert_eq!(mock.get_rollup_call_count(), 1);

            let result2 = backend.get_rollup(layer_id).await.unwrap();
            assert_eq!(result2, Some(rollup_id));
            assert_eq!(
                mock.get_rollup_call_count(),
                1,
                "second call should hit cache, not disk"
            );
        }

        #[tokio::test]
        async fn get_rollup_caches_none_for_layers_without_rollup() {
            let mock = MockMetadataBackend::new();
            let layer_id = test_layer_id(2);

            let backend = make_lru_backend(mock.clone());

            let result1 = backend.get_rollup(layer_id).await.unwrap();
            assert_eq!(result1, None);
            assert_eq!(mock.get_rollup_call_count(), 1);

            let result2 = backend.get_rollup(layer_id).await.unwrap();
            assert_eq!(result2, None);
            assert_eq!(
                mock.get_rollup_call_count(),
                1,
                "negative result should be cached too"
            );
        }

        #[tokio::test]
        async fn set_rollup_updates_cache() {
            let mock = MockMetadataBackend::new();
            let layer_id = test_layer_id(3);
            let rollup_id = test_layer_id(100);

            let backend = make_lru_backend(mock.clone());

            let result1 = backend.get_rollup(layer_id).await.unwrap();
            assert_eq!(result1, None);
            assert_eq!(mock.get_rollup_call_count(), 1);

            backend.set_rollup(layer_id, rollup_id).await.unwrap();

            let result2 = backend.get_rollup(layer_id).await.unwrap();
            assert_eq!(result2, Some(rollup_id));
            assert_eq!(
                mock.get_rollup_call_count(),
                1,
                "get_rollup after set_rollup should use updated cache"
            );
        }

        #[tokio::test]
        async fn set_rollup_overwrites_previous_cached_value() {
            let mock = MockMetadataBackend::new();
            let layer_id = test_layer_id(4);
            let rollup_v1 = test_layer_id(200);
            let rollup_v2 = test_layer_id(201);
            mock.rollup_map.write().unwrap().insert(layer_id, rollup_v1);

            let backend = make_lru_backend(mock.clone());

            let result1 = backend.get_rollup(layer_id).await.unwrap();
            assert_eq!(result1, Some(rollup_v1));

            backend.set_rollup(layer_id, rollup_v2).await.unwrap();

            let result2 = backend.get_rollup(layer_id).await.unwrap();
            assert_eq!(result2, Some(rollup_v2));
            assert_eq!(
                mock.get_rollup_call_count(),
                1,
                "only the initial get_rollup should hit the mock backend"
            );
        }

        #[tokio::test]
        async fn cloned_backend_shares_rollup_cache() {
            let mock = MockMetadataBackend::new();
            let layer_id = test_layer_id(5);
            let rollup_id = test_layer_id(300);
            mock.rollup_map.write().unwrap().insert(layer_id, rollup_id);

            let backend1 = make_lru_backend(mock.clone());
            let backend2 = backend1.clone();

            let result1 = backend1.get_rollup(layer_id).await.unwrap();
            assert_eq!(result1, Some(rollup_id));
            assert_eq!(mock.get_rollup_call_count(), 1);

            let result2 = backend2.get_rollup(layer_id).await.unwrap();
            assert_eq!(result2, Some(rollup_id));
            assert_eq!(
                mock.get_rollup_call_count(),
                1,
                "cloned backend should share the rollup cache"
            );
        }

        /// Documents the interaction between CachedLayerStore::register_rollup
        /// and LruArchiveBackend's rollup metadata cache.
        ///
        /// When CachedLayerStore::register_rollup is called, it:
        ///   1. Delegates to ArchiveLayerStore::register_rollup which calls
        ///      write_rollup_file → set_rollup on LruArchiveBackend, updating
        ///      both disk and the rollup metadata cache.
        ///   2. Invalidates the layer object cache (LockingHashMapLayerCache)
        ///      for the entire parent chain so stale InternalLayer objects are
        ///      evicted.
        ///
        /// When get_layer is subsequently called (after invalidation), it must
        /// call get_rollup to check whether the layer has a rollup. With the
        /// rollup metadata cache, this lookup is served from memory — no disk
        /// I/O for rollup metadata.
        ///
        /// This test models that exact flow and verifies that the rollup
        /// metadata cache survives layer object cache invalidation.
        #[tokio::test]
        async fn register_rollup_flow_caches_metadata_for_layer_rediscovery() {
            let mock = MockMetadataBackend::new();
            let layer_id = test_layer_id(10);
            let rollup_id = test_layer_id(500);

            let backend = make_lru_backend(mock.clone());

            // Phase 1: Initial layer load — no rollup exists yet.
            // get_layer_with_cache calls get_rollup to check for rollup metadata.
            let result = backend.get_rollup(layer_id).await.unwrap();
            assert_eq!(result, None);
            assert_eq!(mock.get_rollup_call_count(), 1);

            // Phase 2: register_rollup writes the rollup via set_rollup.
            // This updates the underlying storage AND the rollup metadata cache.
            backend.set_rollup(layer_id, rollup_id).await.unwrap();

            // Phase 3: CachedLayerStore invalidates the layer object cache
            // (LockingHashMapLayerCache). This is external to LruArchiveBackend
            // — it only evicts InternalLayer objects, not rollup metadata.
            // The rollup metadata cache in LruArchiveBackend is NOT affected.

            // Phase 4: get_layer is called again after layer cache invalidation.
            // It needs get_rollup to rediscover the rollup. The rollup metadata
            // cache serves the result without any backend call.
            let result = backend.get_rollup(layer_id).await.unwrap();
            assert_eq!(result, Some(rollup_id));
            assert_eq!(
                mock.get_rollup_call_count(),
                1,
                "after register_rollup + layer cache invalidation, get_rollup \
                 should serve from the rollup metadata cache without hitting \
                 the underlying backend"
            );

            // Verify a different layer still hits the backend (guard against
            // false positives from a cache that returns stale data for all ids).
            let other_layer = test_layer_id(11);
            let _ = backend.get_rollup(other_layer).await.unwrap();
            assert_eq!(
                mock.get_rollup_call_count(),
                2,
                "an uncached layer should still hit the underlying backend"
            );
        }
    }
}
