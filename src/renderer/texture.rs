use std::{fs, io, path::Path, sync::Arc};

use gltf::{buffer::Data, image::Source};
use image::{DynamicImage, GenericImageView, ImageFormat};
use vulkano::{
    device::Queue,
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    sync::GpuFuture,
};

pub struct Texture {
    pub image: Arc<ImageView<Arc<ImmutableImage>>>,
}

impl Texture {
    pub fn from_gltf_texture(
        graphics_queue: &Arc<Queue>,
        base_path: &str,
        image: &gltf::Texture,
        buffers: &Vec<Data>,
    ) -> Texture {
        let image = match image.source().source() {
            Source::View { view, mime_type } => {
                let parent_buffer_data = &buffers[view.buffer().index()].0;
                let begin = view.offset();
                let end = begin + view.length();
                let data = &parent_buffer_data[begin..end];
                match mime_type {
                    "image/jpeg" => image::load_from_memory_with_format(data, ImageFormat::Jpeg),
                    "image/png" => image::load_from_memory_with_format(data, ImageFormat::Png),
                    _ => panic!(format!(
                        "unsupported image type (image: {}, mime_type: {})",
                        image.index(),
                        mime_type
                    )),
                }
            }
            Source::Uri { uri, mime_type } => {
                let base_path = Path::new(base_path);

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
                        _ => panic!(format!(
                            "unsupported image type (image: {}, mime_type: {})",
                            image.index(),
                            mime_type
                        )),
                    }
                } else if let Some(mime_type) = mime_type {
                    let path = base_path
                        .parent()
                        .unwrap_or_else(|| Path::new("./"))
                        .join(uri);

                    println!("loading texture from {}", path.display());

                    let file = fs::File::open(path).unwrap();
                    let reader = io::BufReader::new(file);
                    match mime_type {
                        "image/jpeg" => image::load(reader, ImageFormat::Jpeg),
                        "image/png" => image::load(reader, ImageFormat::Png),
                        _ => panic!(format!(
                            "unsupported image type (image: {}, mime_type: {})",
                            image.index(),
                            mime_type
                        )),
                    }
                } else {
                    let path = base_path
                        .parent()
                        .unwrap_or_else(|| Path::new("./"))
                        .join(uri);

                    println!("loading texture from {}", path.display());

                    image::open(path)
                }
            }
        }
        .unwrap();

        let image = Self::create_image_view(graphics_queue, &image);

        Texture { image }
    }

    pub fn empty(graphics_queue: &Arc<Queue>) -> Texture {
        let image = DynamicImage::new_rgb8(1, 1);
        let view = Self::create_image_view(graphics_queue, &image);

        Texture { image: view }
    }

    fn create_image_view(
        graphics_queue: &Arc<Queue>,
        image: &DynamicImage,
    ) -> Arc<ImageView<Arc<ImmutableImage>>> {
        let width = image.width();
        let height = image.height();

        let dimensions = ImageDimensions::Dim2d {
            width,
            height,
            // TODO: what are array_layers?
            array_layers: 1,
        };

        let image_rgba = image.to_rgba8();

        let (image, future) = ImmutableImage::from_iter(
            image_rgba.into_raw().iter().cloned(),
            dimensions,
            // vulkano already supports mipmap generation so we don't need to do this by hand
            MipmapsCount::Log2,
            Format::R8G8B8A8_SRGB,
            graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        ImageView::new(image).unwrap()
    }
}
