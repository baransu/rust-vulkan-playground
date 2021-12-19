use std::{sync::Arc, time::Instant};

use vulkano::{
    buffer::{BufferUsage, ImmutableBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::Queue,
    format::Format,
    pipeline::{GraphicsPipeline, Pipeline},
    sync::GpuFuture,
};

use super::{context::Context, texture::Texture, vertex::Vertex};

pub struct Primitive {
    pub vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    pub index_buffer: Arc<ImmutableBuffer<[u32]>>,
    pub index_count: u32,

    pub material: usize,
}

impl Primitive {
    fn new(
        context: &Context,
        material: usize,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    ) -> Primitive {
        let vertex_buffer = Self::create_vertex_buffer(&context.graphics_queue, &vertices);
        let index_buffer = Self::create_index_buffer(&context.graphics_queue, &indices);

        Primitive {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            material,
        }
    }

    fn create_vertex_buffer(
        graphics_queue: &Arc<Queue>,
        vertices: &Vec<Vertex>,
    ) -> Arc<ImmutableBuffer<[Vertex]>> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            vertices.iter().cloned(),
            BufferUsage::vertex_buffer(),
            // TODO: idealy it should be transfer queue?
            graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        buffer
    }

    fn create_index_buffer(
        graphics_queue: &Arc<Queue>,
        indices: &Vec<u32>,
    ) -> Arc<ImmutableBuffer<[u32]>> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            indices.iter().cloned(),
            BufferUsage::index_buffer(),
            graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        buffer
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
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        diffuse: Texture,
        metallic_roughness: Texture,
        normal: Texture,
    ) -> Material {
        let descriptor_set = Self::create_descriptor_set(
            context,
            graphics_pipeline,
            &diffuse,
            &metallic_roughness,
            &normal,
        );

        Material {
            diffuse,
            metallic_roughness,
            normal,
            descriptor_set,
        }
    }

    fn create_descriptor_set(
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        diffuse: &Texture,
        metallic_roughness: &Texture,
        normal: &Texture,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(1)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder
            .add_sampled_image(diffuse.image.clone(), context.image_sampler.clone())
            .unwrap();

        set_builder
            .add_sampled_image(normal.image.clone(), context.image_sampler.clone())
            .unwrap();

        set_builder
            .add_sampled_image(
                metallic_roughness.image.clone(),
                context.image_sampler.clone(),
            )
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
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        path: &str,
    ) -> Model {
        let (document, buffers, images) = gltf::import(path).unwrap();

        assert_eq!(
            document.scenes().len(),
            1,
            "Only one scene per file is supported"
        );

        let scene = document.scenes().nth(0).unwrap();

        let id = path.to_string();

        let start_time = Instant::now();
        println!("Loading model: {}", id);

        let primitives: Vec<Primitive> = document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .map(|primitive| {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = &reader.read_positions().unwrap().collect::<Vec<[f32; 3]>>();
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
                        let position = *position;
                        let normal = *normals.get(index).unwrap_or(&[1.0, 1.0, 1.0]);
                        let uv = *tex_coords.get(index).unwrap_or(&[0.0, 0.0]);
                        let tangent = *tangents.get(index).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);

                        Vertex::new(position, normal, uv, tangent)
                    })
                    .collect::<Vec<_>>();

                let indices = reader
                    .read_indices()
                    .map(|indices| indices.into_u32().collect::<Vec<_>>())
                    .unwrap();

                Primitive::new(
                    context,
                    primitive.material().index().unwrap(),
                    vertices,
                    indices,
                )
            })
            .collect();

        let materials: Vec<Material> = document
            .materials()
            .map(|material| {
                let pbr = material.pbr_metallic_roughness();

                let diffuse = pbr.base_color_texture().map(|info| {
                    Texture::from_gltf_texture(
                        &context,
                        path,
                        &info.texture(),
                        &images,
                        Format::R8G8B8A8_UNORM,
                    )
                });

                let normal = material.normal_texture().map(|info| {
                    Texture::from_gltf_texture(
                        &context,
                        path,
                        &info.texture(),
                        &images,
                        Format::R8G8B8A8_UNORM,
                    )
                });

                let metalic_roughness = pbr.metallic_roughness_texture().map(|info| {
                    Texture::from_gltf_texture(
                        &context,
                        path,
                        &info.texture(),
                        &images,
                        Format::R8G8B8A8_UNORM,
                    )
                });

                Material::new(
                    context,
                    graphics_pipeline,
                    // TODO: how to correctly handle optional values?
                    diffuse.unwrap_or_else(|| Texture::empty(&context)),
                    metalic_roughness.unwrap_or_else(|| Texture::empty(&context)),
                    normal.unwrap_or_else(|| Texture::empty(&context)),
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

        println!("Loaded model: {} in {:?}", id, start_time.elapsed());

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
