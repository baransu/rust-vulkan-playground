use env_logger::Env;
use glam::{EulerRot, Quat, Vec3};
use renderer::{entity::Entity, renderer::Renderer, transform::Transform};

pub mod imgui_renderer;
pub mod renderer;

const DAMAGED_HELMET: &str = "res/models/damaged_helmet/scene.gltf";
const SPONZA: &str = "glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf";
const BOTTLE: &str = "glTF-Sample-Models/2.0/WaterBottle/glTF/WaterBottle.gltf";

const PLANE: &str = "res/models/plane/plane.gltf";

const MODEL_PATHS: [&str; 2] = [
    // DAMAGED_HELMET,
    BOTTLE,
    // "res/models/cube/cube.gltf",
    // "res/models/sphere/sphere.gltf",
    PLANE,
    // SPONZA,
    // "glTF-Sample-Models/2.0/WaterBottle/glTF/WaterBottle.gltf",
];

// const SKYBOX_PATH: &str = "res/hdr/uffizi_cube.ktx";
const SKYBOX_PATH: &str = "res/hdr/je_gray_park_4k.pic";
// const SKYBOX_PATH: &str = "res/hdr/pisa_cube.ktx";

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();

    puffin::set_scopes_on(true);

    let mut app = Renderer::initialize(MODEL_PATHS.to_vec(), SKYBOX_PATH);

    // let count = 10;
    // let start = -(count / 2);
    // let end = count / 2;

    // // Helmets
    // for x in start..end {
    //     for z in start..end {
    //         let translation = Vec3::new(x as f32 * 3.0, 3.0, z as f32 * 3.0);
    //         let material = Material {
    //             // diffuse: Vec3::new(
    //             //     rng.gen_range(0.0..1.0),
    //             //     rng.gen_range(0.0..1.0),
    //             //     rng.gen_range(0.0..1.0),
    //             // ),
    //             ..Default::default()
    //         };

    //         let entity = Entity::new(
    //             "damaged_helmet",
    //             Transform {
    //                 translation,
    //                 rotation: Quat::from_euler(EulerRot::XYZ, 90.0_f32.to_radians(), 0.0, 0.0),
    //                 scale: Vec3::ONE,
    //             },
    //             material,
    //         );

    //         scene.add_entity(entity);
    //     }
    // }

    // scene.add_entity(Entity::new(
    //     "Sphere",
    //     Transform {
    //         translation: Vec3::ZERO,
    //         rotation: Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, 0.0),
    //         scale: Vec3::ONE,
    //     },
    //     Default::default(),
    // ));

    app.add_entity(Entity::new(
        DAMAGED_HELMET,
        Transform {
            translation: Vec3::new(0.0, 0.0, 1.0),
            rotation: Quat::from_euler(EulerRot::XYZ, 90.0_f32.to_radians(), 0.0, 0.0),
            scale: Vec3::ONE,
        },
    ));

    app.add_entity(Entity::new(
        BOTTLE,
        Transform {
            translation: Vec3::new(-2.0, 0.0, 0.0),
            rotation: Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, 0.0),
            scale: Vec3::ONE * 5.0,
        },
    ));

    // Plane
    app.add_entity(Entity::new(
        PLANE,
        Transform {
            rotation: Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, 0.0),
            scale: Vec3::ONE * 5.0,
            translation: Vec3::new(0.0, -2.0, 0.0),
        },
    ));

    // scene.add_entity(Entity::new(
    //     PLANE,
    //     Transform {
    //         rotation: Quat::from_euler(EulerRot::XYZ, 180.0_f32.to_radians(), 0.0, 0.0),
    //         scale: Vec3::ONE * 5.0,
    //         translation: Vec3::new(0.0, 5.0, 0.0),
    //     },
    // ));

    // sponza
    app.add_entity(Entity::new(
        SPONZA,
        Transform {
            rotation: Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, 0.0),
            scale: Vec3::ONE * 0.01,
            translation: Vec3::new(0.0, 0.0, 0.0),
        },
    ));

    // // point light cubes for reference
    // for light in scene.point_lights.clone() {
    //     let transform = Transform {
    //         rotation: Quat::IDENTITY,
    //         scale: Vec3::ONE * 0.2,
    //         translation: light.position,
    //     };

    //     scene.add_entity(Entity::new("Cube", transform));
    // }

    // // dir light
    // scene.add_entity(Entity::new(
    //     "Cube",
    //     Transform {
    //         rotation: Quat::IDENTITY,
    //         scale: Vec3::ONE * 0.2,
    //         translation: Scene::dir_light_position(),
    //     },
    // ));

    app.main_loop();
}
