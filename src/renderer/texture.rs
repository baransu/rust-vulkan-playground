use std::{
    fs::{self, File},
    io::{self, BufReader},
    path::Path,
    sync::Arc,
};

use gltf::image::Source;
use image::{DynamicImage, GenericImageView, ImageFormat};
use ktx::{Decoder, KtxInfo};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer},
    device::Queue,
    format::Format,
    image::{
        immutable::SubImage,
        view::{ImageView, ImageViewType},
        ImageCreateFlags, ImageDimensions, ImageLayout, ImageUsage, ImmutableImage, MipmapsCount,
    },
    sync::GpuFuture,
};

use super::context::Context;

pub struct Texture {
    pub image: Arc<ImageView<ImmutableImage>>,
}

impl Texture {
    pub fn from_ktx(context: &Context, image: Decoder<BufReader<File>>) -> Texture {
        let width = image.pixel_width();
        let height = image.pixel_height();

        println!("Loading cubemap texture: {}x{}", width, height);

        let image_rgba = image.read_textures().next().unwrap().to_vec();

        let dimensions = ImageDimensions::Dim2d {
            width,
            height,
            // TODO: what are array_layers?
            array_layers: 6,
        };

        let format = Format::R16G16B16A16_SFLOAT;

        // Most of this logic is copied from vulkano immutable.rs from_buffer method
        let (image, future) = {
            let usage = ImageUsage {
                transfer_destination: true,
                transfer_source: false,
                sampled: true,
                ..ImageUsage::none()
            };

            let flags = ImageCreateFlags {
                cube_compatible: true,
                ..ImageCreateFlags::none()
            };

            let layout = ImageLayout::ShaderReadOnlyOptimal;

            let source = CpuAccessibleBuffer::from_iter(
                context.device.clone(),
                BufferUsage::transfer_source(),
                false,
                image_rgba,
            )
            .unwrap();

            let (image, initializer) = ImmutableImage::uninitialized(
                context.device.clone(),
                dimensions,
                format,
                MipmapsCount::One,
                usage,
                flags,
                layout,
                context.device.active_queue_families(),
            )
            .unwrap();

            let init = SubImage::new(
                Arc::new(initializer),
                0,
                1,
                0,
                1,
                ImageLayout::ShaderReadOnlyOptimal,
            );

            let mut cbb = AutoCommandBufferBuilder::primary(
                context.device.clone(),
                context.graphics_queue.family(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            cbb.copy_buffer_to_image_dimensions(
                source,
                init,
                [0, 0, 0],
                dimensions.width_height_depth(),
                0,
                dimensions.array_layers(),
                0,
            )
            .unwrap();

            let cb = cbb.build().unwrap();

            let future = cb.execute(context.graphics_queue.clone()).unwrap();

            (image, future)
        };

        future.flush().unwrap();

        Texture {
            image: ImageView::start(image)
                .with_type(ImageViewType::Cube)
                .build()
                .unwrap(),
        }
    }

    pub fn from_gltf_texture(
        queue: &Arc<Queue>,
        base_path: &str,
        image: &gltf::Texture,
        images: &Vec<gltf::image::Data>,
        format: Format,
    ) -> Texture {
        let image = match image.source().source() {
            Source::View { view, mime_type } => {
                let data = &images[view.buffer().index()].pixels;

                println!("Loading Source::View texture");

                match mime_type {
                    "image/jpeg" => image::load_from_memory_with_format(data, ImageFormat::Jpeg),
                    "image/png" => image::load_from_memory_with_format(data, ImageFormat::Png),
                    _ => panic!(
                        "unsupported image type (image: {}, mime_type: {})",
                        image.index(),
                        mime_type
                    ),
                }
            }
            Source::Uri { uri, mime_type } => {
                let base_path = Path::new(base_path);

                println!("Loading Source::Uri({}, {:?}) texture", uri, mime_type);

                if uri.starts_with("data:") {
                    let encoded = uri.split(',').nth(1).unwrap();
                    let data = base64::decode(&encoded).unwrap();
                    let mime_type = if let Some(ty) = mime_type {
                        ty
                    } else {
                        uri.split(',')
                            .nth(0)
                            .unwrap()
                            .split(':')
                            .nth(1)
                            .unwrap()
                            .split(';')
                            .nth(0)
                            .unwrap()
                    };

                    match mime_type {
                        "image/jpeg" => {
                            image::load_from_memory_with_format(&data, ImageFormat::Jpeg)
                        }
                        "image/png" => image::load_from_memory_with_format(&data, ImageFormat::Png),
                        _ => panic!(
                            "unsupported image type (image: {}, mime_type: {})",
                            image.index(),
                            mime_type
                        ),
                    }
                } else if let Some(mime_type) = mime_type {
                    let path = base_path
                        .parent()
                        .unwrap_or_else(|| Path::new("./"))
                        .join(uri);

                    let file = fs::File::open(path).unwrap();
                    let reader = io::BufReader::new(file);
                    match mime_type {
                        "image/jpeg" => image::load(reader, ImageFormat::Jpeg),
                        "image/png" => image::load(reader, ImageFormat::Png),
                        _ => panic!(
                            "unsupported image type (image: {}, mime_type: {})",
                            image.index(),
                            mime_type
                        ),
                    }
                } else {
                    let path = base_path
                        .parent()
                        .unwrap_or_else(|| Path::new("./"))
                        .join(uri);

                    image::open(path)
                }
            }
        }
        .unwrap();

        let image = Self::create_image_view(queue, &image, format);

        Texture { image }
    }

    pub fn empty(queue: &Arc<Queue>) -> Texture {
        let image = DynamicImage::new_rgb8(1, 1);
        let view = Self::create_image_view(queue, &image, Format::R8G8B8A8_UNORM);

        Texture { image: view }
    }

    pub fn create_image_view(
        queue: &Arc<Queue>,
        image: &DynamicImage,
        format: Format,
    ) -> Arc<ImageView<ImmutableImage>> {
        let width = image.width();
        let height = image.height();

        let dimensions = ImageDimensions::Dim2d {
            width,
            height,
            // TODO: what are array_layers?
            array_layers: 1,
        };

        let (image, future) = if format == Format::R16G16B16A16_SFLOAT {
            ImmutableImage::from_iter(
                image.to_rgba16().into_raw().iter().cloned(),
                dimensions,
                // vulkano already supports mipmap generation so we don't need to do this by hand
                MipmapsCount::Log2,
                format,
                queue.clone(),
            )
            .unwrap()
        } else {
            ImmutableImage::from_iter(
                image.to_rgba8().into_raw().iter().cloned(),
                dimensions,
                // vulkano already supports mipmap generation so we don't need to do this by hand
                MipmapsCount::Log2,
                format,
                queue.clone(),
            )
            .unwrap()
        };

        future.flush().unwrap();

        ImageView::new(image).unwrap()
    }
}
