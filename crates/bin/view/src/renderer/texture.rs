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
        let index = image.source().index();

        let image = match image.source().source() {
            Source::View { .. } => {
                let data = &images[index];
                let width = data.width;
                let height = data.height;

                log::debug!("Loading Source::View({:?}) texture", data.format);

                let size = (width * height) as usize;
                let mut buffer = Vec::new();
                for i in 0..size {
                    let rgba = match data.format {
                        gltf::image::Format::R8 => [data.pixels[i], 0, 0, std::u8::MAX],
                        gltf::image::Format::R8G8 => [
                            data.pixels[i * 2 + 0],
                            data.pixels[i * 2 + 1],
                            0,
                            std::u8::MAX,
                        ],
                        gltf::image::Format::R8G8B8 => [
                            data.pixels[i * 3 + 0],
                            data.pixels[i * 3 + 1],
                            data.pixels[i * 3 + 2],
                            std::u8::MAX,
                        ],
                        gltf::image::Format::R8G8B8A8 => [
                            data.pixels[i * 4 + 0],
                            data.pixels[i * 4 + 1],
                            data.pixels[i * 4 + 2],
                            data.pixels[i * 4 + 3],
                        ],
                        format => panic!(
                            "unsupported image format (image: {}, format: {:?})",
                            image.index(),
                            format
                        ),
                    };

                    buffer.extend_from_slice(&rgba);
                }

                Self::image_view_from_rgba8_buffer(queue, width, height, &buffer, format, mipmaps)
            }
            Source::Uri { uri, mime_type } => {
                let base_path = Path::new(base_path);

                log::debug!("Loading Source::Uri({}, {:?}) texture", uri, mime_type);

                let image = if uri.starts_with("data:") {
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
                    .unwrap()
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
                    .unwrap()
                } else {
                    let path = base_path
                        .parent()
                        .unwrap_or_else(|| Path::new("./"))
                        .join(uri);

                    image::open(path).unwrap()
                };

                Self::image_view_from_dynamic_image(queue, &image, format, mipmaps)
            }
        };

        Texture { image }
    }

    pub fn empty(queue: &Arc<Queue>) -> Texture {
        let image = DynamicImage::new_rgb8(1, 1);
        let view = Self::image_view_from_dynamic_image(
            queue,
            &image,
            Format::R8G8B8A8_UNORM,
            MipmapsCount::One,
        );

        Texture { image: view }
    }

    pub fn image_view_from_dynamic_image(
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

    pub fn image_view_from_rgba8_buffer(
        queue: &Arc<Queue>,
        width: u32,
        height: u32,
        buffer: &Vec<u8>,
        format: Format,
        mipmaps: MipmapsCount,
    ) -> Arc<ImageView<ImmutableImage>> {
        let dimensions = ImageDimensions::Dim2d {
            width,
            height,
            array_layers: 1,
        };

        let (image, future) = ImmutableImage::from_iter(
            buffer.iter().cloned(),
            dimensions,
            mipmaps,
            format,
            queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        ImageView::new(image).unwrap()
    }
}
