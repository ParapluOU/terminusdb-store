//! In-memory implementation of storage traits.

use std::collections::HashMap;
use std::io;

use async_trait::async_trait;

use super::consts::FILENAMES;
use super::file::*;
use super::label::*;
use super::layer::*;

pub use tdb_succinct::storage::memory::*;

#[derive(Clone, Default)]
pub struct MemoryLayerStore {
    layers: futures_locks::RwLock<HashMap<[u32; 5], HashMap<String, MemoryBackedStore>>>,
}

impl MemoryLayerStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl PersistentLayerStore for MemoryLayerStore {
    type File = MemoryBackedStore;

    async fn directories(&self) -> io::Result<Vec<[u32; 5]>> {
        let guard = self.layers.read().await;
        Ok(guard.keys().cloned().collect())
    }

    async fn create_named_directory(&self, name: [u32; 5]) -> io::Result<[u32; 5]> {
        let mut guard = self.layers.write().await;
        if guard.contains_key(&name) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "layer already exists",
            ));
        }
        guard.insert(name, HashMap::new());

        Ok(name)
    }

    async fn directory_exists(&self, name: [u32; 5]) -> io::Result<bool> {
        let guard = self.layers.read().await;
        Ok(guard.contains_key(&name))
    }

    async fn file_exists(&self, directory: [u32; 5], file: &str) -> io::Result<bool> {
        let guard = self.layers.read().await;
        if let Some(files) = guard.get(&directory) {
            if let Some(file) = files.get(file) {
                file.exists().await
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    async fn get_file(&self, directory: [u32; 5], name: &str) -> io::Result<Self::File> {
        let guard = self.layers.read().await;
        if let Some(files) = guard.get(&directory) {
            if let Some(file) = files.get(name) {
                Ok(file.clone())
            } else {
                std::mem::drop(guard); // release read lock cause it is time to write
                let mut guard = self.layers.write().await;
                let files = guard.get_mut(&directory).unwrap();
                let file = MemoryBackedStore::new();
                let result = file.clone();
                files.insert(name.to_string(), file);
                Ok(result)
            }
        } else {
            Err(io::Error::new(io::ErrorKind::NotFound, "layer not found"))
        }
    }

    async fn abort_directory(&self, directory: [u32; 5]) -> io::Result<()> {
        self.layers.write().await.remove(&directory);
        Ok(())
    }

    async fn publish_imported_layers_shared(
        &self,
        layers: Vec<([u32; 5], HashMap<String, bytes::Bytes>)>,
    ) -> io::Result<()> {
        let mut ids = std::collections::HashSet::with_capacity(layers.len());
        if layers.iter().any(|(id, _)| !ids.insert(*id)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "duplicate imported layer id",
            ));
        }
        let mut prepared = Vec::with_capacity(layers.len());
        for (id, files) in layers {
            let mut prepared_files = HashMap::with_capacity(files.len());
            for (name, bytes) in files {
                let file = MemoryBackedStore::new();
                let mut writer = file.open_write().await?;
                tokio::io::AsyncWriteExt::write_all(&mut writer, &bytes).await?;
                tokio::io::AsyncWriteExt::flush(&mut writer).await?;
                writer.sync_all().await?;
                prepared_files.insert(name, file);
            }
            prepared.push((id, prepared_files));
        }
        let mut guard = self.layers.write().await;
        if prepared.iter().any(|(id, _)| guard.contains_key(id)) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "refusing to publish over an existing layer",
            ));
        }
        for (id, files) in prepared {
            guard.insert(id, files);
        }
        Ok(())
    }

    async fn write_proof_state_bytes(&self, directory: [u32; 5], bytes: &[u8]) -> io::Result<()> {
        let file = MemoryBackedStore::new();
        let mut writer = file.open_write().await?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, bytes).await?;
        tokio::io::AsyncWriteExt::flush(&mut writer).await?;
        writer.sync_all().await?;

        let mut guard = self.layers.write().await;
        let files = guard
            .get_mut(&directory)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "layer not found"))?;
        if files.contains_key(FILENAMES.proof_state) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "proof state already exists",
            ));
        }
        files.insert(FILENAMES.proof_state.to_string(), file);
        Ok(())
    }

    async fn replace_proof_state_bytes_if_matches(
        &self,
        directory: [u32; 5],
        expected: &[u8],
        replacement: &[u8],
    ) -> io::Result<()> {
        let file = MemoryBackedStore::new();
        let mut writer = file.open_write().await?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, replacement).await?;
        tokio::io::AsyncWriteExt::flush(&mut writer).await?;
        writer.sync_all().await?;
        let mut guard = self.layers.write().await;
        let files = guard
            .get_mut(&directory)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "layer not found"))?;
        let current = files
            .get(FILENAMES.proof_state)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "proof state not found"))?;
        if current.map().await?.as_ref() != expected {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "proof state changed during migration",
            ));
        }
        files.insert(FILENAMES.proof_state.to_string(), file);
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct MemoryLabelStore {
    labels: futures_locks::RwLock<HashMap<String, Label>>,
}

impl MemoryLabelStore {
    pub fn new() -> MemoryLabelStore {
        Default::default()
    }
}

#[async_trait]
impl LabelStore for MemoryLabelStore {
    async fn labels(&self) -> io::Result<Vec<Label>> {
        let labels = self.labels.read().await;
        Ok(labels.values().cloned().collect())
    }

    async fn create_label(&self, name: &str) -> io::Result<Label> {
        let label = Label::new_empty(name);

        let mut labels = self.labels.write().await;
        if labels.get(&label.name).is_some() {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "label already exists",
            ))
        } else {
            labels.insert(label.name.clone(), label.clone());
            Ok(label)
        }
    }

    async fn get_label(&self, name: &str) -> io::Result<Option<Label>> {
        let name = name.to_owned();
        let labels = self.labels.read().await;
        Ok(labels.get(&name).cloned())
    }

    async fn set_label_option(
        &self,
        label: &Label,
        layer: Option<[u32; 5]>,
    ) -> io::Result<Option<Label>> {
        let new_label = label.with_updated_layer(layer);

        let mut labels = self.labels.write().await;

        match labels.get(&new_label.name) {
            None => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "label does not exist",
            )),
            Some(old_label) => {
                if old_label.version + 1 != new_label.version {
                    Ok(None)
                } else {
                    labels.insert(new_label.name.clone(), new_label.clone());

                    Ok(Some(new_label))
                }
            }
        }
    }

    async fn delete_label(&self, name: &str) -> io::Result<bool> {
        let mut labels = self.labels.write().await;

        Ok(labels.remove(name).is_some())
    }
}

#[cfg(test)]
pub fn base_layer_memory_files() -> BaseLayerFiles<MemoryBackedStore> {
    BaseLayerFiles {
        node_dictionary_files: DictionaryFiles {
            blocks_file: MemoryBackedStore::new(),
            offsets_file: MemoryBackedStore::new(),
        },
        predicate_dictionary_files: DictionaryFiles {
            blocks_file: MemoryBackedStore::new(),
            offsets_file: MemoryBackedStore::new(),
        },
        value_dictionary_files: TypedDictionaryFiles {
            types_present_file: MemoryBackedStore::new(),
            type_offsets_file: MemoryBackedStore::new(),
            blocks_file: MemoryBackedStore::new(),
            offsets_file: MemoryBackedStore::new(),
        },

        id_map_files: IdMapFiles {
            node_value_idmap_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            predicate_idmap_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
        },

        subjects_file: MemoryBackedStore::new(),
        objects_file: MemoryBackedStore::new(),

        s_p_adjacency_list_files: AdjacencyListFiles {
            bitindex_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            nums_file: MemoryBackedStore::new(),
        },
        sp_o_adjacency_list_files: AdjacencyListFiles {
            bitindex_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            nums_file: MemoryBackedStore::new(),
        },
        o_ps_adjacency_list_files: AdjacencyListFiles {
            bitindex_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            nums_file: MemoryBackedStore::new(),
        },
        predicate_wavelet_tree_files: BitIndexFiles {
            bits_file: MemoryBackedStore::new(),
            blocks_file: MemoryBackedStore::new(),
            sblocks_file: MemoryBackedStore::new(),
        },
    }
}

#[cfg(test)]
pub fn child_layer_memory_files() -> ChildLayerFiles<MemoryBackedStore> {
    ChildLayerFiles {
        node_dictionary_files: DictionaryFiles {
            blocks_file: MemoryBackedStore::new(),
            offsets_file: MemoryBackedStore::new(),
        },
        predicate_dictionary_files: DictionaryFiles {
            blocks_file: MemoryBackedStore::new(),
            offsets_file: MemoryBackedStore::new(),
        },
        value_dictionary_files: TypedDictionaryFiles {
            types_present_file: MemoryBackedStore::new(),
            type_offsets_file: MemoryBackedStore::new(),
            blocks_file: MemoryBackedStore::new(),
            offsets_file: MemoryBackedStore::new(),
        },

        id_map_files: IdMapFiles {
            node_value_idmap_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            predicate_idmap_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
        },

        pos_subjects_file: MemoryBackedStore::new(),
        pos_objects_file: MemoryBackedStore::new(),
        neg_subjects_file: MemoryBackedStore::new(),
        neg_objects_file: MemoryBackedStore::new(),

        pos_s_p_adjacency_list_files: AdjacencyListFiles {
            bitindex_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            nums_file: MemoryBackedStore::new(),
        },
        pos_sp_o_adjacency_list_files: AdjacencyListFiles {
            bitindex_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            nums_file: MemoryBackedStore::new(),
        },
        pos_o_ps_adjacency_list_files: AdjacencyListFiles {
            bitindex_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            nums_file: MemoryBackedStore::new(),
        },
        neg_s_p_adjacency_list_files: AdjacencyListFiles {
            bitindex_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            nums_file: MemoryBackedStore::new(),
        },
        neg_sp_o_adjacency_list_files: AdjacencyListFiles {
            bitindex_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            nums_file: MemoryBackedStore::new(),
        },
        neg_o_ps_adjacency_list_files: AdjacencyListFiles {
            bitindex_files: BitIndexFiles {
                bits_file: MemoryBackedStore::new(),
                blocks_file: MemoryBackedStore::new(),
                sblocks_file: MemoryBackedStore::new(),
            },
            nums_file: MemoryBackedStore::new(),
        },
        pos_predicate_wavelet_tree_files: BitIndexFiles {
            bits_file: MemoryBackedStore::new(),
            blocks_file: MemoryBackedStore::new(),
            sblocks_file: MemoryBackedStore::new(),
        },
        neg_predicate_wavelet_tree_files: BitIndexFiles {
            bits_file: MemoryBackedStore::new(),
            blocks_file: MemoryBackedStore::new(),
            sblocks_file: MemoryBackedStore::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn write_and_read_memory_backed() {
        let file = MemoryBackedStore::new();

        let mut w = file.open_write().await.unwrap();
        w.write_all(&[1, 2, 3]).await.unwrap();
        w.sync_all().await.unwrap();
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
    async fn write_and_map_memory_backed() {
        let file = MemoryBackedStore::new();

        let mut w = file.open_write().await.unwrap();
        w.write_all(&[1, 2, 3]).await.unwrap();
        w.sync_all().await.unwrap();
        let map = file.map().await.unwrap();

        assert_eq!(vec![1, 2, 3], map.as_ref());
    }

    #[tokio::test]
    async fn create_layers_from_memory_store() {
        let store = MemoryLayerStore::new();
        let mut builder = store.create_base_layer().await.unwrap();
        let base_name = builder.name();

        builder.add_value_triple(ValueTriple::new_string_value("cow", "says", "moo"));
        builder.add_value_triple(ValueTriple::new_string_value("pig", "says", "oink"));
        builder.add_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));

        builder.commit_boxed().await.unwrap();
        builder = store.create_child_layer(base_name).await.unwrap();
        let child_name = builder.name();

        builder.remove_value_triple(ValueTriple::new_string_value("duck", "says", "quack"));
        builder.add_value_triple(ValueTriple::new_node("cow", "likes", "pig"));

        builder.commit_boxed().await.unwrap();
        let layer = store.get_layer(child_name).await.unwrap().unwrap();

        assert!(layer.value_triple_exists(&ValueTriple::new_string_value("cow", "says", "moo")));
        assert!(layer.value_triple_exists(&ValueTriple::new_string_value("pig", "says", "oink")));
        assert!(layer.value_triple_exists(&ValueTriple::new_node("cow", "likes", "pig")));
        assert!(!layer.value_triple_exists(&ValueTriple::new_string_value("duck", "says", "quack")));
    }

    #[tokio::test]
    async fn memory_proof_publication_is_single_assignment() {
        let store = MemoryLayerStore::new();
        let id = store.create_directory().await.unwrap();
        store.write_proof_state_bytes(id, b"first").await.unwrap();
        assert_eq!(
            store
                .write_proof_state_bytes(id, b"second")
                .await
                .unwrap_err()
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(
            store
                .get_file(id, FILENAMES.proof_state)
                .await
                .unwrap()
                .map()
                .await
                .unwrap()
                .as_ref(),
            b"first"
        );
    }

    #[tokio::test]
    async fn memory_batch_rejects_duplicate_ids_without_publication() {
        let store = MemoryLayerStore::new();
        let id = [1, 2, 3, 4, 5];
        let mut first = HashMap::new();
        first.insert("a".to_string(), vec![1]);
        let mut second = HashMap::new();
        second.insert("b".to_string(), vec![2]);
        assert_eq!(
            store
                .publish_imported_layers(vec![(id, first), (id, second)])
                .await
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert!(!store.directory_exists(id).await.unwrap());
    }

    #[tokio::test]
    async fn memory_create_and_retrieve_equal_label() {
        let store = MemoryLabelStore::new();
        let foo = store.create_label("foo").await.unwrap();
        assert_eq!(foo, store.get_label("foo").await.unwrap().unwrap());
    }

    #[tokio::test]
    async fn memory_update_label_succeeds() {
        let store = MemoryLabelStore::new();
        let foo = store.create_label("foo").await.unwrap();

        assert_eq!(
            1,
            store
                .set_label(&foo, [6, 7, 8, 9, 10])
                .await
                .unwrap()
                .unwrap()
                .version
        );

        assert_eq!(1, store.get_label("foo").await.unwrap().unwrap().version);
    }

    #[tokio::test]
    async fn memory_update_label_twice_from_same_label_object_fails() {
        let store = MemoryLabelStore::new();
        let foo = store.create_label("foo").await.unwrap();

        assert!(store
            .set_label(&foo, [6, 7, 8, 9, 10])
            .await
            .unwrap()
            .is_some());
        assert!(store
            .set_label(&foo, [1, 1, 1, 1, 1])
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn memory_update_label_twice_from_updated_label_object_succeeds() {
        let store = MemoryLabelStore::new();
        let foo = store.create_label("foo").await.unwrap();

        let foo2 = store
            .set_label(&foo, [6, 7, 8, 9, 10])
            .await
            .unwrap()
            .unwrap();
        assert!(store
            .set_label(&foo2, [1, 1, 1, 1, 1])
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn create_and_delete_label() {
        let store = MemoryLabelStore::new();

        store.create_label("foo").await.unwrap();
        assert!(store.get_label("foo").await.unwrap().is_some());
        assert!(store.delete_label("foo").await.unwrap());
        assert!(store.get_label("foo").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_nonexistent_label() {
        let store = MemoryLabelStore::new();

        assert!(!store.delete_label("foo").await.unwrap());
    }
}
