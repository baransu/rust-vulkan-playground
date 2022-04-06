use std::{sync::Arc, time::Instant};

use vulkano::{
    buffer::{BufferUsage, ImmutableBuffer},
    descriptor_set::{layout::DescriptorSetLayout, PersistentDescriptorSet},
    device::Queue,
    format::Format,
    image::MipmapsCount,
    sync::GpuFuture,
};

use super::{shaders::Vertex, texture::Texture};

pub struct Primitive {
    _vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    pub index_buffer: Arc<ImmutableBuffer<[u32]>>,
    pub descriptor_set: Arc<PersistentDescriptorSet>,
    pub index_count: u32,
    pub material: usize,
}

impl Primitive {
    fn new(
        queue: &Arc<Queue>,
        layout: &Arc<DescriptorSetLayout>,
        material: usize,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    ) -> Primitive {
        let vertex_buffer = Self::create_vertex_buffer(queue, &vertices);
        let index_buffer = Self::create_index_buffer(queue, &indices);
        let descriptor_set = Self::create_descriptor_set(layout, &vertex_buffer);

        Primitive {
            _vertex_buffer: vertex_buffer,
            index_buffer,
            descriptor_set,
            index_count: indices.len() as u32,
            material,
        }
    }

    fn create_vertex_buffer(
        queue: &Arc<Queue>,
        vertices: &Vec<Vertex>,
    ) -> Arc<ImmutableBuffer<[Vertex]>> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            vertices.iter().cloned(),
            BufferUsage::storage_buffer(),
            queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        buffer
    }

    fn create_index_buffer(queue: &Arc<Queue>, indices: &Vec<u32>) -> Arc<ImmutableBuffer<[u32]>> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            indices.iter().cloned(),
            BufferUsage::index_buffer(),
            queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        buffer
    }

    fn create_descriptor_set(
        layout: &Arc<DescriptorSetLayout>,
        vertex_buffer: &Arc<ImmutableBuffer<[Vertex]>>,
    ) -> Arc<PersistentDescriptorSet> {
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder.add_buffer(vertex_buffer.clone()).unwrap();

        set_builder.build().unwrap()
    }
}

pub struct Material {
    pub diffuse: Texture,
    pub metallic_roughness: Texture,
    pub normal: Texture,

    pub descriptor_set: Arc<PersistentDescriptorSet>,
}

impl Material {
    fn new(
        layout: &Arc<DescriptorSetLayout>,
        diffuse: Texture,
        metallic_roughness: Texture,
        normal: Texture,
    ) -> Material {
        let descriptor_set =
            Self::create_descriptor_set(layout, &diffuse, &metallic_roughness, &normal);

        Material {
            diffuse,
            metallic_roughness,
            normal,
            descriptor_set,
        }
    }

    fn create_descriptor_set(
        layout: &Arc<DescriptorSetLayout>,
        diffuse: &Texture,
        metallic_roughness: &Texture,
        normal: &Texture,
    ) -> Arc<PersistentDescriptorSet> {
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder.add_image(diffuse.image.clone()).unwrap();

        set_builder.add_image(normal.image.clone()).unwrap();

        set_builder
            .add_image(metallic_roughness.image.clone())
            .unwrap();

        set_builder.build().unwrap()
    }
}

pub struct Mesh {
    pub primitives: Vec<usize>,
}

pub struct Node {
    pub mesh: Option<usize>,
    pub children: Vec<usize>,
    // TODO: add local transform
}

pub struct Model {
    pub id: String,

    pub primitives: Vec<Primitive>,
    pub materials: Vec<Material>,

    pub meshes: Vec<Mesh>,

    pub nodes: Vec<Node>,

    pub root_nodes: Vec<usize>,
}

impl Model {
    pub fn load_gltf(
        queue: &Arc<Queue>,
        path: &str,
        vertices_layout: &Arc<DescriptorSetLayout>,
        materials_layout: &Arc<DescriptorSetLayout>,
    ) -> Model {
        let start_time = Instant::now();
        let id = path.to_string();
        log::debug!("Loading model: {}", id);

        let (document, buffers, images) = gltf::import(path).unwrap();

        let scene = document.default_scene().unwrap();

        let primitives: Vec<Primitive> = document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .map(|primitive| {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = &reader.read_positions().unwrap().collect::<Vec<_>>();
                let normals = &reader
                    .read_normals()
                    .map_or(vec![], |normals| normals.collect());

                let tangents = &reader
                    .read_tangents()
                    .map_or(vec![], |tangets| tangets.collect());

                // TODO: why gltf has more than one uv channel?
                let tex_coords = &reader
                    .read_tex_coords(0)
                    .map_or(vec![], |coords| coords.into_f32().collect());

                let vertices = positions
                    .iter()
                    .enumerate()
                    .map(|(index, position)| {
                        let [px, py, pz] = *position;
                        let [nx, ny, nz] = *normals.get(index).unwrap_or(&[1.0, 1.0, 1.0]);
                        let [ux, uy] = *tex_coords.get(index).unwrap_or(&[0.0, 0.0]);
                        let [tx, ty, tz, tw] =
                            *tangents.get(index).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);

                        Vertex {
                            px,
                            py,
                            pz,
                            nx,
                            ny,
                            nz,
                            ux,
                            uy,
                            tx,
                            ty,
                            tz,
                            tw,
                        }
                    })
                    .collect::<Vec<_>>();

                let indices = reader
                    .read_indices()
                    .map(|indices| indices.into_u32().collect::<Vec<_>>())
                    .unwrap();

                let (_vertex_count, remap) =
                    meshopt::generate_vertex_remap(&vertices, Some(&indices));

                let final_vertices =
                    meshopt::remap_vertex_buffer(&vertices, vertices.len(), &remap);
                let final_indices =
                    meshopt::remap_index_buffer(Some(&indices), indices.len(), &remap);

                Primitive::new(
                    queue,
                    vertices_layout,
                    primitive.material().index().unwrap(),
                    final_vertices,
                    final_indices,
                )
            })
            .collect();

        let materials: Vec<Material> = document
            .materials()
            .map(|material| {
                let pbr = material.pbr_metallic_roughness();

                let diffuse = pbr.base_color_texture().map(|info| {
                    Texture::from_gltf_texture(
                        &queue,
                        path,
                        &info.texture(),
                        &images,
                        Format::R8G8B8A8_UNORM,
                        MipmapsCount::Log2,
                    )
                });

                let normal = material.normal_texture().map(|info| {
                    Texture::from_gltf_texture(
                        &queue,
                        path,
                        &info.texture(),
                        &images,
                        Format::R8G8B8A8_UNORM,
                        MipmapsCount::One,
                    )
                });

                let metalic_roughness = pbr.metallic_roughness_texture().map(|info| {
                    Texture::from_gltf_texture(
                        &queue,
                        path,
                        &info.texture(),
                        &images,
                        Format::R8G8B8A8_UNORM,
                        MipmapsCount::One,
                    )
                });

                Material::new(
                    materials_layout,
                    // TODO: how to correctly handle optional values?
                    diffuse.unwrap_or_else(|| Texture::empty(&queue)),
                    metalic_roughness.unwrap_or_else(|| Texture::empty(&queue)),
                    normal.unwrap_or_else(|| Texture::empty(&queue)),
                )
            })
            .collect();

        let meshes: Vec<Mesh> = document
            .meshes()
            .map(|mesh| {
                let primitives = mesh
                    .primitives()
                    .map(|primitive| primitive.index())
                    .collect();

                Mesh { primitives }
            })
            .collect();

        let nodes: Vec<Node> = document
            .nodes()
            .map(|node| {
                let mesh = node.mesh().map(|mesh| mesh.index());

                let children = node.children().map(|child| child.index()).collect();

                Node { mesh, children }
            })
            .collect();

        let root_nodes = scene.nodes().map(|node| node.index()).collect();

        log::info!("Loaded model: {} in {:?}", id, start_time.elapsed());

        Model {
            id,
            primitives,
            materials,
            meshes,
            nodes,
            root_nodes,
        }
    }
}
