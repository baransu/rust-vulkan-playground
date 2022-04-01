use std::{fs, io, path::Path, sync::Arc};

use gltf::image::Source;
use image::{DynamicImage, ImageFormat};
use vulkano::{
    device::Queue,
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    sync::GpuFuture,
};

pub struct Texture {
    pub image: Arc<ImageView<ImmutableImage>>,
}

impl Texture {
    pub fn from_gltf_texture(
        queue: &Arc<Queue>,
        base_path: &str,
        image: &gltf::Texture,
        images: &Vec<gltf::image::Data>,
        format: Format,
        mipmaps: MipmapsCount,
    ) -> Texture {
        let image = match image.source().source() {
            Source::View { view, mime_type } => {
                let data = &images[view.buffer().index()].pixels;

                log::debug!("Loading Source::View({}) texture", mime_type);

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

                log::debug!("Loading Source::Uri({}, {:?}) texture", uri, mime_type);

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

        let image = Self::create_image_view(queue, &image, format, mipmaps);

        Texture { image }
    }

    pub fn empty(queue: &Arc<Queue>) -> Texture {
        let image = DynamicImage::new_rgb8(1, 1);
        let view =
            Self::create_image_view(queue, &image, Format::R8G8B8A8_UNORM, MipmapsCount::One);

        Texture { image: view }
    }

    pub fn create_image_view(
        queue: &Arc<Queue>,
        image: &DynamicImage,
        format: Format,
        mipmaps: MipmapsCount,
    ) -> Arc<ImageView<ImmutableImage>> {
        let width = image.width();
        let height = image.height();

        let dimensions = ImageDimensions::Dim2d {
            width,
            height,
            array_layers: 1,
        };

        let (image, future) = if format == Format::R16G16B16A16_SFLOAT {
            ImmutableImage::from_iter(
                image.to_rgba16().into_raw().iter().cloned(),
                dimensions,
                mipmaps,
                format,
                queue.clone(),
            )
            .unwrap()
        } else {
            ImmutableImage::from_iter(
                image.to_rgba8().into_raw().iter().cloned(),
                dimensions,
                mipmaps,
                format,
                queue.clone(),
            )
            .unwrap()
        };

        future.flush().unwrap();

        ImageView::new(image).unwrap()
    }
}
