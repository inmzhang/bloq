//! glTF geometry and standalone viewer export.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::sync::Arc;

use base64::{Engine as _, prelude::BASE64_STANDARD};
use bloq_utils::{Basis, Pauli, RGBA, UDirection};
use glam::{IVec3, Vec3, vec3};
use gltf_json::{
    Accessor, Asset, Buffer, Material, Root, Value,
    accessor::{ComponentType, GenericComponentType, Type},
    buffer::{Target, View},
    extensions::material::{Material as MaterialExtensions, Unlit},
    material::{AlphaMode, PbrBaseColorFactor, PbrMetallicRoughness, StrengthFactor},
    mesh::{Mode, Primitive, Semantic},
    validation::Checked::Valid,
};

use crate::{
    Block, BlockGraphError, BlockKind, Direction, PatchRotationKind, Pipe, Stabilizer,
    StabilizerGenerator, StabilizerRowKind, WalkingKind, ZXGraph, graph::BlockGraph,
};

/// A positioned text label carried alongside the geometry.
#[derive(Debug, Clone)]
pub struct TextLabel {
    /// World-space position of the label.
    pub position: Vec3,
    /// Label text.
    pub text: String,
    /// Label color.
    pub color: RGBA,
}

/// Intermediate geometry for glTF export: colored triangles, colored line
/// segments, and text labels, accumulated before serialization to a glTF model.
#[derive(Debug, Clone, Default)]
pub struct GltfData {
    /// Filled triangles grouped by color.
    pub triangles: BTreeMap<RGBA, Vec<[Vec3; 3]>>,
    /// Line segments grouped by color.
    pub lines: BTreeMap<RGBA, Vec<[Vec3; 2]>>,
    /// Positioned text labels.
    pub texts: Vec<TextLabel>,
    /// Face colors displayed without diffuse shading, for module ownership.
    pub unlit_colors: BTreeSet<RGBA>,
}

// Each color's triangle ranges retain definition ownership without copying geometry.
pub(crate) type ModuleFaceRanges = BTreeMap<RGBA, Vec<(Option<usize>, Range<usize>)>>;
type ModuleMaterial = (usize, usize, [f32; 4], [f32; 4], &'static str);

/// Selects filled block-graph faces to remove from a glTF export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GltfFaceSelector {
    /// Every face with this outward normal.
    All(Direction),
    /// One face of the block at `position`.
    Block {
        /// Anchor position of the selected block.
        position: IVec3,
        /// Outward normal of the selected face.
        face: Direction,
    },
    /// One face of the pipe between `u` and `v`.
    Pipe {
        /// First pipe endpoint.
        u: IVec3,
        /// Second pipe endpoint.
        v: IVec3,
        /// Outward normal of the selected face.
        face: Direction,
    },
}

impl GltfFaceSelector {
    fn for_block(self, position: IVec3) -> Option<Direction> {
        match self {
            Self::All(face) => Some(face),
            Self::Block { position: p, face } if p == position => Some(face),
            _ => None,
        }
    }

    fn for_pipe(self, u: IVec3, v: IVec3) -> Option<Direction> {
        match self {
            Self::All(face) => Some(face),
            Self::Pipe { u: a, v: b, face } if (a == u && b == v) || (a == v && b == u) => {
                Some(face)
            }
            _ => None,
        }
    }
}

/// Viewport background for the standalone HTML viewer.
const VIEWER_BACKGROUND: &str = "#D8E0E6";

/// Ambient brightness shared by the editor and its baked glTF lighting.
pub const EDITOR_AMBIENT_BRIGHTNESS: f32 = 1065.0;
/// Diagonal key-light illuminance shared by the editor and its baked glTF lighting.
pub const EDITOR_DIRECTIONAL_ILLUMINANCE: f32 = 1055.0;

fn editor_face_brightness(triangle: &[Vec3; 3]) -> f32 {
    let [a, b, c] = *triangle;
    let normal = (b - a).cross(c - a).normalize_or_zero();
    let exposure = 2.0_f32.powf(-9.7) / 1.2;
    exposure
        * (0.4524 * EDITOR_AMBIENT_BRIGHTNESS
            + 0.3255 * EDITOR_DIRECTIONAL_ILLUMINANCE * normal.dot(Vec3::ONE.normalize()).max(0.0))
}

fn linear_palette(color: RGBA) -> [f32; 4] {
    let mut channels = color.to_f32_array();
    for channel in &mut channels[..3] {
        *channel = if *channel <= 0.04045 {
            *channel / 12.92
        } else {
            ((*channel + 0.055) / 1.055).powf(2.4)
        };
    }
    channels
}

pub(crate) const POPPED_X_FACE_COLOR: RGBA = RGBA::from_hex(0xA61E2D, 255);
pub(crate) const POPPED_Y_FACE_COLOR: RGBA = RGBA::from_hex(0x188038, 255);
pub(crate) const POPPED_Z_FACE_COLOR: RGBA = RGBA::from_hex(0x174EA6, 255);

// Fully diffuse: any metallic or specular response makes faces of the same
// basis read as different colors depending on how they face the light.
const DEFAULT_METALLIC_FACTOR: f32 = 0.0;
const DEFAULT_ROUGHNESS_FACTOR: f32 = 1.0;

#[derive(Debug, Clone, Copy)]
enum FaceStyle {
    Fill,
    Outline,
    FillAndOutline,
}

impl FaceStyle {
    fn has_fill(self) -> bool {
        matches!(self, FaceStyle::Fill | FaceStyle::FillAndOutline)
    }

    fn has_outline(self) -> bool {
        matches!(self, FaceStyle::Outline | FaceStyle::FillAndOutline)
    }
}

impl GltfData {
    fn color_module_faces(&mut self, color: RGBA) {
        let triangles: Vec<_> = std::mem::take(&mut self.triangles)
            .into_values()
            .flatten()
            .collect();
        if !triangles.is_empty() {
            self.triangles.insert(color, triangles);
            self.unlit_colors.insert(color);
        }
    }

    /// Returns a copy with every triangle color's alpha multiplied by `scale`
    /// (clamped to opaque). Lines and labels are unchanged.
    pub fn scale_face_opacity(&self, scale: f32) -> GltfData {
        let apply_scale = |color: RGBA| RGBA {
            r: color.r,
            g: color.g,
            b: color.b,
            a: ((color.a as f32) * scale) as u8,
        };

        // Scaling can collapse two distinct colors onto the same RGBA (e.g. when
        // alpha saturates), so merge into the same key instead of letting the
        // later entry overwrite the earlier one's triangles.
        let mut triangles: BTreeMap<RGBA, Vec<[Vec3; 3]>> = BTreeMap::new();
        for (color, tris) in &self.triangles {
            triangles
                .entry(apply_scale(*color))
                .or_default()
                .extend(tris.iter().copied());
        }

        GltfData {
            triangles,
            lines: self.lines.clone(),
            texts: self.texts.clone(),
            unlit_colors: self.unlit_colors.iter().copied().map(apply_scale).collect(),
        }
    }

    /// Removes filled faces whose outward normal matches one of `directions`.
    ///
    /// Line geometry is retained so the opened graph keeps its silhouette.
    pub fn pop_faces_at_directions(mut self, directions: &[Direction]) -> GltfData {
        if directions.is_empty() {
            return self;
        }

        self.triangles.retain(|_, triangles| {
            triangles.retain(|[a, b, c]| {
                let normal = (*b - *a).cross(*c - *a).normalize_or_zero();
                !directions
                    .iter()
                    .any(|direction| normal.dot(direction.to_vec3()) > 0.999)
            });
            !triangles.is_empty()
        });
        self
    }

    /// Merges another data set into this one, combining geometry of matching colors.
    pub fn extend(&mut self, other: GltfData) {
        self.unlit_colors.extend(other.unlit_colors);
        for (color, triangles) in other.triangles {
            self.add_triangles(color, triangles);
        }
        for (color, lines) in other.lines {
            self.add_line(color, lines);
        }
        self.texts.extend(other.texts);
    }

    fn add_triangles(&mut self, color: RGBA, triangles: impl IntoIterator<Item = [Vec3; 3]>) {
        self.triangles.entry(color).or_default().extend(triangles);
    }

    fn add_line(&mut self, color: RGBA, lines: impl IntoIterator<Item = [Vec3; 2]>) {
        self.lines.entry(color).or_default().extend(lines);
    }

    fn add_face(&mut self, center: Vec3, side1: Vec3, side2: Vec3, rgba: RGBA, style: FaceStyle) {
        let origin = center - (side1 + side2) * 0.5;
        if style.has_fill() {
            self.add_triangles(
                rgba,
                [
                    [origin, origin + side1, origin + side2],
                    [origin + side1 + side2, origin + side2, origin + side1],
                ],
            );
        }
        if style.has_outline() {
            self.add_line(
                RGBA::LINE_BLACK,
                [
                    [origin, origin + side1],
                    [origin + side1 + side2, origin + side1],
                    [origin, origin + side2],
                    [origin + side1 + side2, origin + side2],
                ],
            );
        }
    }

    fn add_cube(&mut self, center: Vec3, sides: Vec3, rgba: RGBA, style: FaceStyle) {
        let [x, y, z] = center.to_array();
        let [sx, sy, sz] = sides.to_array();
        let mut f = |c, s1, s2| {
            self.add_face(c, s1, s2, rgba, style);
        };
        f(
            vec3(x, y, z - sz * 0.5),
            vec3(0.0, sy, 0.0),
            vec3(sx, 0.0, 0.0),
        );
        f(
            vec3(x, y, z + sz * 0.5),
            vec3(sx, 0.0, 0.0),
            vec3(0.0, sy, 0.0),
        );
        f(
            vec3(x, y - sy * 0.5, z),
            vec3(sx, 0.0, 0.0),
            vec3(0.0, 0.0, sz),
        );
        f(
            vec3(x, y + sy * 0.5, z),
            vec3(0.0, 0.0, sz),
            vec3(sx, 0.0, 0.0),
        );
        f(
            vec3(x - sx * 0.5, y, z),
            vec3(0.0, 0.0, sz),
            vec3(0.0, sy, 0.0),
        );
        f(
            vec3(x + sx * 0.5, y, z),
            vec3(0.0, sy, 0.0),
            vec3(0.0, 0.0, sz),
        );
    }

    fn add_prism_face(&mut self, corners: [Vec3; 4], rgba: RGBA, style: FaceStyle) {
        if style.has_fill() {
            self.add_triangles(
                rgba,
                [
                    [corners[0], corners[1], corners[2]],
                    [corners[2], corners[3], corners[0]],
                ],
            );
        }
        if style.has_outline() {
            self.add_line(
                RGBA::LINE_BLACK,
                [
                    [corners[0], corners[1]],
                    [corners[1], corners[2]],
                    [corners[2], corners[3]],
                    [corners[3], corners[0]],
                ],
            );
        }
    }

    fn add_double_color_cube(&mut self, center: Vec3, sides: Vec3, rgba1: RGBA, rgba2: RGBA) {
        self.add_cube(center, sides, RGBA::LINE_BLACK, FaceStyle::Outline);
        let [x, y, z] = center.to_array();
        let [sx, sy, sz] = (0.5 * sides).to_array();
        self.add_triangles(
            rgba1,
            [
                [
                    vec3(x - sx, y - sy, z - sz),
                    vec3(x - sx, y - sy, z + sz),
                    vec3(x - sx, y + sy, z + sz),
                ],
                [
                    vec3(x - sx, y - sy, z - sz),
                    vec3(x + sx, y - sy, z - sz),
                    vec3(x + sx, y - sy, z + sz),
                ],
                [
                    vec3(x - sx, y - sy, z + sz),
                    vec3(x + sx, y - sy, z + sz),
                    vec3(x + sx, y + sy, z + sz),
                ],
                [
                    vec3(x + sx, y - sy, z - sz),
                    vec3(x + sx, y + sy, z - sz),
                    vec3(x + sx, y + sy, z + sz),
                ],
                [
                    vec3(x - sx, y + sy, z - sz),
                    vec3(x - sx, y + sy, z + sz),
                    vec3(x + sx, y + sy, z + sz),
                ],
                [
                    vec3(x - sx, y - sy, z - sz),
                    vec3(x - sx, y + sy, z - sz),
                    vec3(x + sx, y + sy, z - sz),
                ],
            ],
        );
        self.add_triangles(
            rgba2,
            [
                [
                    vec3(x - sx, y + sy, z + sz),
                    vec3(x - sx, y + sy, z - sz),
                    vec3(x - sx, y - sy, z - sz),
                ],
                [
                    vec3(x + sx, y - sy, z + sz),
                    vec3(x - sx, y - sy, z + sz),
                    vec3(x - sx, y - sy, z - sz),
                ],
                [
                    vec3(x + sx, y + sy, z + sz),
                    vec3(x - sx, y + sy, z + sz),
                    vec3(x - sx, y - sy, z + sz),
                ],
                [
                    vec3(x + sx, y + sy, z + sz),
                    vec3(x + sx, y - sy, z + sz),
                    vec3(x + sx, y - sy, z - sz),
                ],
                [
                    vec3(x + sx, y + sy, z + sz),
                    vec3(x + sx, y + sy, z - sz),
                    vec3(x - sx, y + sy, z - sz),
                ],
                [
                    vec3(x + sx, y + sy, z - sz),
                    vec3(x + sx, y - sy, z - sz),
                    vec3(x - sx, y - sy, z - sz),
                ],
            ],
        );
    }

    /// Swaps the `y` and `z` axes of all geometry, converting from the graph's
    /// time-up (`z`) convention to the exporter's y-up convention.
    ///
    /// Triangle winding is flipped to keep front faces oriented after the swap.
    pub fn swap_yz(self) -> GltfData {
        let mut transformed = GltfData {
            unlit_colors: self.unlit_colors,
            ..Default::default()
        };
        let swap = |v: Vec3| vec3(v.x, v.z, v.y);

        for (color, triangles) in self.triangles {
            let transformed_triangles = triangles
                .into_iter()
                .map(|[a, b, c]| [swap(a), swap(c), swap(b)])
                .collect();
            transformed.triangles.insert(color, transformed_triangles);
        }

        for (color, lines) in self.lines {
            let transformed_lines = lines.into_iter().map(|[a, b]| [swap(a), swap(b)]).collect();
            transformed.lines.insert(color, transformed_lines);
        }
        transformed.texts = self
            .texts
            .into_iter()
            .map(|label| TextLabel {
                position: swap(label.position),
                ..label
            })
            .collect();

        transformed
    }

    fn to_buffer_bytes(&self) -> Vec<u8> {
        let positions = self
            .triangles
            .values()
            .flatten()
            .flatten()
            .chain(self.lines.values().flatten().flatten())
            .flat_map(Vec3::to_array);
        // Baking the editor's diffuse light keeps exports independent of the
        // viewing application's environment and specular-light defaults.
        // ponytail: double-sided backs keep the front's baked shade; live
        // lighting is needed if viewing opened geometry from inside must match.
        let colors = self.triangles.iter().flat_map(|(color, triangles)| {
            triangles.iter().flat_map(|triangle| {
                let brightness = if self.unlit_colors.contains(color) {
                    1.0
                } else {
                    editor_face_brightness(triangle)
                };
                [[brightness, brightness, brightness, 1.0]; 3]
                    .into_iter()
                    .flatten()
            })
        });
        positions.chain(colors).flat_map(f32::to_le_bytes).collect()
    }

    /// Applies `f` to every vertex, line endpoint, and text-label position.
    ///
    /// Triangle winding is preserved as-is; this does not re-orient normals, so
    /// only use it for orientation-preserving transforms such as translations,
    /// rotations, and positive scales.
    pub fn map_points(mut self, f: impl Fn(Vec3) -> Vec3) -> GltfData {
        for triangles in self.triangles.values_mut() {
            for triangle in triangles {
                for point in triangle {
                    *point = f(*point);
                }
            }
        }

        for lines in self.lines.values_mut() {
            for line in lines {
                for point in line {
                    *point = f(*point);
                }
            }
        }

        for label in &mut self.texts {
            label.position = f(label.position);
        }

        self
    }

    /// Builds a glTF [`Root`] document from the accumulated geometry.
    fn to_model(&self) -> Root {
        let mut root = gltf_json::Root {
            asset: Asset {
                version: "2.0".to_string(),
                copyright: Some("Generated by `bloq_graph` crate".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        root.extensions_used.push("KHR_materials_unlit".to_string());
        root.extensions_required
            .push("KHR_materials_unlit".to_string());

        let mut create_material = |color: RGBA, double_sided: bool| {
            root.push(Material {
                pbr_metallic_roughness: PbrMetallicRoughness {
                    // glTF factors use linear RGB; the authored palette uses sRGB.
                    base_color_factor: PbrBaseColorFactor(linear_palette(color)),
                    metallic_factor: StrengthFactor(DEFAULT_METALLIC_FACTOR),
                    roughness_factor: StrengthFactor(DEFAULT_ROUGHNESS_FACTOR),
                    ..Default::default()
                },
                double_sided,
                extensions: Some(MaterialExtensions {
                    unlit: Some(Unlit {}),
                }),
                alpha_mode: Valid(if color.a < 255 {
                    AlphaMode::Blend
                } else {
                    AlphaMode::Opaque
                }),
                ..Default::default()
            })
        };

        let tri_materials: BTreeMap<_, _> = self
            .triangles
            .keys()
            .map(|&c| (c, create_material(c, true)))
            .collect();
        let line_materials: BTreeMap<_, _> = self
            .lines
            .keys()
            .map(|&c| (c, create_material(c, false)))
            .collect();

        let buffer_bytes = self.to_buffer_bytes();
        let buffer_index = root.push(Buffer {
            uri: Some(format!(
                "data:application/octet-stream;base64,{}",
                BASE64_STANDARD.encode(&buffer_bytes)
            )),
            byte_length: buffer_bytes.len().into(),
            extensions: None,
            extras: Default::default(),
            name: None,
        });

        let mut byte_offset = 0;
        let mut color_byte_offset = (self.triangles.values().map(Vec::len).sum::<usize>() * 3
            + self.lines.values().map(Vec::len).sum::<usize>() * 2)
            * 3
            * 4;
        let mut primitives = Vec::new();

        let mut add_primitive = |count: usize,
                                 min: Vec3,
                                 max: Vec3,
                                 material_index: gltf_json::Index<Material>,
                                 mode: Mode| {
            let byte_length = count * 3 * 4;
            let view_index = root.push(View {
                buffer: buffer_index,
                byte_length: byte_length.into(),
                byte_offset: Some(byte_offset.into()),
                byte_stride: None,
                target: Some(Valid(Target::ArrayBuffer)),
                extensions: None,
                extras: Default::default(),
                name: None,
            });

            let accessor_index = root.push(Accessor {
                buffer_view: Some(view_index),
                byte_offset: Some(0usize.into()),
                count: count.into(),
                component_type: Valid(GenericComponentType(ComponentType::F32)),
                type_: Valid(Type::Vec3),
                normalized: false,
                min: Some(Value::from(Vec::from(min.to_array()))),
                max: Some(Value::from(Vec::from(max.to_array()))),
                sparse: None,
                extensions: None,
                extras: Default::default(),
                name: None,
            });

            let mut attributes = BTreeMap::new();
            attributes.insert(Valid(Semantic::Positions), accessor_index);
            if mode == Mode::Triangles {
                let color_byte_length = count * 4 * 4;
                let color_view = root.push(View {
                    buffer: buffer_index,
                    byte_length: color_byte_length.into(),
                    byte_offset: Some(color_byte_offset.into()),
                    target: Some(Valid(Target::ArrayBuffer)),
                    byte_stride: None,
                    extensions: None,
                    extras: Default::default(),
                    name: None,
                });
                let color_accessor = root.push(Accessor {
                    buffer_view: Some(color_view),
                    count: count.into(),
                    component_type: Valid(GenericComponentType(ComponentType::F32)),
                    type_: Valid(Type::Vec4),
                    byte_offset: Some(0usize.into()),
                    normalized: false,
                    min: None,
                    max: None,
                    sparse: None,
                    extensions: None,
                    extras: Default::default(),
                    name: None,
                });
                attributes.insert(Valid(Semantic::Colors(0)), color_accessor);
                color_byte_offset += color_byte_length;
            }
            primitives.push(Primitive {
                attributes,
                indices: None,
                material: Some(material_index),
                mode: Valid(mode),
                targets: None,
                extensions: None,
                extras: Default::default(),
            });

            byte_offset += byte_length;
        };

        for (color, triangles) in &self.triangles {
            let count = triangles.len() * 3;
            let (min, max) = bounds(triangles.iter().flatten());
            add_primitive(count, min, max, tri_materials[color], Mode::Triangles);
        }

        for (color, lines) in &self.lines {
            let count = lines.len() * 2;
            let (min, max) = bounds(lines.iter().flatten());
            add_primitive(count, min, max, line_materials[color], Mode::Lines);
        }

        let mesh_index = root.push(gltf_json::Mesh {
            primitives,
            weights: None,
            extensions: None,
            extras: Default::default(),
            name: None,
        });
        let node_index = root.push(gltf_json::Node {
            mesh: Some(mesh_index),
            ..Default::default()
        });
        root.push(gltf_json::Scene {
            nodes: vec![node_index],
            extensions: None,
            extras: Default::default(),
            name: None,
        });
        root
    }

    fn module_model(&self, ranges: &ModuleFaceRanges) -> (Root, Vec<ModuleMaterial>) {
        let mut model = self.to_model();
        let original_primitives = std::mem::take(&mut model.meshes[0].primitives);
        let mut materials = BTreeMap::new();
        let mut toggles = Vec::new();
        for ((color, triangles), primitive) in self.triangles.iter().zip(&original_primitives) {
            for (owner, range) in &ranges[color] {
                let mut part = primitive.clone();
                // Accessors select slices of the existing position and shaded color buffers.
                for (semantic, bytes_per_vertex) in
                    [(Semantic::Positions, 12), (Semantic::Colors(0), 16)]
                {
                    let mut accessor =
                        model.accessors[part.attributes[&Valid(semantic.clone())].value()].clone();
                    accessor.byte_offset = Some((range.start * 3 * bytes_per_vertex).into());
                    accessor.count = (range.len() * 3).into();
                    if semantic == Semantic::Positions {
                        let (min, max) = bounds(triangles[range.clone()].iter().flatten());
                        accessor.min = Some(Value::from(Vec::from(min.to_array())));
                        accessor.max = Some(Value::from(Vec::from(max.to_array())));
                    }
                    part.attributes
                        .insert(Valid(semantic), model.push(accessor));
                }
                if let Some(owner) = owner {
                    let material = *materials.entry((*owner, *color)).or_insert_with(|| {
                        let mut material = model.materials[primitive
                            .material
                            .expect("triangle primitive has a material")
                            .value()]
                        .clone();
                        let original = material.pbr_metallic_roughness.base_color_factor.0;
                        let highlight = linear_palette(crate::module_color(*owner));
                        let alpha = if color.a < 255 { "BLEND" } else { "OPAQUE" };
                        material.pbr_metallic_roughness.base_color_factor =
                            PbrBaseColorFactor(highlight);
                        material.alpha_mode = Valid(AlphaMode::Opaque);
                        let index = model.push(material);
                        toggles.push((*owner, index.value(), original, highlight, alpha));
                        index
                    });
                    part.material = Some(material);
                }
                model.meshes[0].primitives.push(part);
            }
        }
        // Outlines and cross-definition pipes retain their ordinary materials.
        model.meshes[0]
            .primitives
            .extend(original_primitives.into_iter().skip(self.triangles.len()));
        (model, toggles)
    }

    /// Writes the scene to a `.gltf` file.
    ///
    /// Vertex colors bake the editor's diffuse response into unlit materials.
    /// Lighting stays fixed; the host viewer does not relight the graph.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::Io`] if the file cannot be written, or
    /// [`BlockGraphError::GltfSerialize`] if serialization fails.
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), BlockGraphError> {
        let path = path.as_ref();
        let model = self.to_model();
        let file = std::fs::File::create(path).map_err(|e| BlockGraphError::Io {
            path: path.to_path_buf(),
            source: Arc::new(e),
        })?;
        gltf_json::serialize::to_writer(file, &model)
            .map_err(|e| BlockGraphError::GltfSerialize(Arc::new(e)))
    }

    /// Renders an HTML page with the glTF scene embedded as a base64 data URI.
    ///
    /// Vertex colors bake the editor's diffuse response into unlit materials.
    /// Lighting stays fixed; the viewer does not relight the graph.
    /// The `model-viewer` renderer is pulled from a CDN, so the page needs
    /// network access to display; the model data itself is inlined.
    ///
    /// # Errors
    ///
    /// Returns [`BlockGraphError::GltfSerialize`] if glTF serialization fails.
    pub fn to_html_str(&self) -> Result<String, BlockGraphError> {
        self.to_html_str_with_legend("")
    }

    pub(crate) fn to_html_str_with_legend(&self, legend: &str) -> Result<String, BlockGraphError> {
        Self::model_html(&self.to_model(), legend, &[])
    }

    pub(crate) fn to_module_html_str(
        &self,
        legend: &str,
        ranges: &ModuleFaceRanges,
    ) -> Result<String, BlockGraphError> {
        let (model, toggles) = self.module_model(ranges);
        Self::model_html(&model, legend, &toggles)
    }

    fn model_html(
        model: &Root,
        legend: &str,
        toggles: &[ModuleMaterial],
    ) -> Result<String, BlockGraphError> {
        let bytes = gltf_json::serialize::to_vec(model)
            .map_err(|error| BlockGraphError::GltfSerialize(Arc::new(error)))?;
        let toggles = gltf_json::serialize::to_string(toggles)
            .map_err(|error| BlockGraphError::GltfSerialize(Arc::new(error)))?;
        let model_data_uri = format!(
            "data:model/gltf+json;base64,{}",
            BASE64_STANDARD.encode(&bytes)
        );
        Ok(format!(
            r##"<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script id="viewer-runtime" type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.2.0/model-viewer.min.js" onerror="window.viewerLoadFailed = true"></script>
  <style>
    body {{ margin: 0; font: 14px system-ui, sans-serif; }}
    model-viewer {{ width: 100vw; height: 100vh; background-color: {VIEWER_BACKGROUND}; }}
    .controls {{ position: absolute; top: 12px; left: 12px; right: 12px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 12px; color: #1f2937; pointer-events: none; }}
    .controls button {{ pointer-events: auto; }}
    .module-legend {{ position: absolute; top: 56px; right: 12px; max-width: 45%; max-height: calc(100vh - 150px); overflow: auto; padding: 8px; border-radius: 4px; background: #f8fafcee; color: #1f2937; font-size: 12px; }}
    .module-legend ul {{ padding: 0; margin: 8px 0 0; list-style: none; }}
    .module-legend li {{ margin: 6px 0; overflow-wrap: anywhere; }}
    .module-legend label {{ cursor: pointer; }}
    .module-legend input {{ vertical-align: middle; margin: 0 5px 0 0; }}
    .module-legend i {{ display: inline-block; width: 10px; height: 10px; margin-right: 5px; }}
    #axes {{ position: absolute; bottom: 12px; left: 12px; width: 100px; height: 100px; pointer-events: none; visibility: hidden; }}
    #axes text {{ font: bold 13px system-ui, sans-serif; text-anchor: middle; dominant-baseline: central; paint-order: stroke; stroke: {VIEWER_BACKGROUND}; stroke-width: 3px; }}
    #status {{ position: absolute; inset: 0; display: grid; place-items: center; padding: 24px; text-align: center; color: #526070; pointer-events: none; }}
    button {{ padding: 6px 12px; border: 1px solid #94a3b2; border-radius: 4px; background: #f8fafc; color: #1f2937; cursor: pointer; }}
  </style>
</head>
<body>
  <model-viewer
    src="{}"
    alt="Block Graph 3D Model"
    camera-controls
    camera-orbit="45deg 54.7356deg auto"
    field-of-view="45deg"
    max-field-of-view="45deg"
    shadow-intensity="0"
    tone-mapping="none"
    exposure="1.0"
    interaction-prompt="none"
    touch-action="none">
  </model-viewer>
  <div id="status" role="status">Loading 3D graph…</div>
  <div class="controls">
    <button type="button" disabled aria-label="Reset graph view">Fit</button>
    <span>Drag to orbit · Right-drag to pan · Scroll to zoom</span>
  </div>
  {legend}
  <svg id="axes" viewBox="0 0 100 100" role="img" aria-label="Graph axes: X and Y are spatial, Z is time">
    <circle cx="50" cy="50" r="3" fill="#526070" />
    <g data-axis="X" stroke="#dc2626" fill="#dc2626"><line x1="50" y1="50" stroke-width="2"/><text stroke-width="0">X</text></g>
    <g data-axis="Y" stroke="#16a34a" fill="#16a34a"><line x1="50" y1="50" stroke-width="2"/><text stroke-width="0">Y</text></g>
    <g data-axis="Z" stroke="#2563eb" fill="#2563eb"><line x1="50" y1="50" stroke-width="2"/><text stroke-width="0">Z</text></g>
  </svg>
  <script>
    const viewer = document.querySelector('model-viewer');
    const fit = document.querySelector('button');
    const status = document.querySelector('#status');
    const axes = document.querySelector('#axes');
    const moduleMaterials = {toggles};
    const moduleToggles = document.querySelectorAll('.module-legend input');
    for (const toggle of moduleToggles) {{
      toggle.addEventListener('change', () => {{
        for (const [owner, index, original, highlight, alpha] of moduleMaterials) {{
          if (owner !== Number(toggle.dataset.module)) continue;
          const material = viewer.model.materials[index];
          material.pbrMetallicRoughness.setBaseColorFactor(toggle.checked ? highlight : original);
          material.setAlphaMode(toggle.checked ? 'OPAQUE' : alpha);
        }}
      }});
    }}
    const updateAxes = () => {{
      const {{theta, phi}} = viewer.getCameraOrbit();
      axes.style.visibility = 'visible';
      // Graph (x,y,z) maps to renderer (x,z,-y), just as in Bloq Editor.
      const directions = {{
        X: [Math.cos(theta), Math.sin(theta) * Math.cos(phi)],
        Y: [Math.sin(theta), -Math.cos(theta) * Math.cos(phi)],
        Z: [0, -Math.sin(phi)]
      }};
      for (const group of axes.querySelectorAll('[data-axis]')) {{
        const [x, y] = directions[group.dataset.axis];
        const line = group.querySelector('line');
        line.setAttribute('x2', 50 + 32 * x);
        line.setAttribute('y2', 50 + 32 * y);
        const label = group.querySelector('text');
        label.setAttribute('x', 50 + 43 * x);
        label.setAttribute('y', 50 + 43 * y);
      }}
    }};
    viewer.addEventListener('camera-change', updateAxes);
    const showError = () => {{ status.textContent = 'Unable to load the 3D viewer. Check your network connection.'; }};
    document.querySelector('#viewer-runtime').addEventListener('error', showError);
    if (window.viewerLoadFailed) showError();
    viewer.addEventListener('error', showError);
    viewer.addEventListener('load', () => {{
      fit.disabled = false;
      for (const toggle of moduleToggles) toggle.disabled = false;
      status.textContent = '';
      updateAxes();
    }});
    fit.addEventListener('click', () => {{
      viewer.cameraTarget = 'auto auto auto';
      viewer.cameraOrbit = '45deg 54.7356deg auto';
      viewer.fieldOfView = '45deg';
    }});
  </script>
</body>
</html>"##,
            model_data_uri
        ))
    }
}

const HADAMARD_STRIP_WIDTH: f32 = 0.2;

fn bounds<'a>(points: impl Iterator<Item = &'a Vec3>) -> (Vec3, Vec3) {
    points.fold((Vec3::MAX, Vec3::MIN), |(min, max), v| {
        (min.min(*v), max.max(*v))
    })
}

fn next_axis(u: UDirection) -> UDirection {
    match u {
        UDirection::X => UDirection::Y,
        UDirection::Y => UDirection::Z,
        UDirection::Z => UDirection::X,
    }
}

fn get_basis(dir: UDirection, kind: crate::CubeKind) -> Basis {
    match dir {
        UDirection::X => kind.x(),
        UDirection::Y => kind.y(),
        UDirection::Z => kind.z(),
    }
}

fn walking_as_gltf_data(kind: WalkingKind, pipe_length: f32) -> GltfData {
    let mut data = GltfData::default();
    let stride = pipe_length + 1.0;
    let movement = kind.movement_3d().as_vec3() * stride;
    let bottom_center = Vec3::NEG_Z * 0.5;
    let top_center = movement + Vec3::Z * 0.5;
    let x = Vec3::X * 0.5;
    let y = Vec3::Y * 0.5;

    let b00 = bottom_center - x - y;
    let b10 = bottom_center + x - y;
    let b11 = bottom_center + x + y;
    let b01 = bottom_center - x + y;
    let t00 = top_center - x - y;
    let t10 = top_center + x - y;
    let t11 = top_center + x + y;
    let t01 = top_center - x + y;

    let boundary = kind.boundary();
    data.add_prism_face(
        [b00, b01, b11, b10],
        boundary.z().into(),
        FaceStyle::FillAndOutline,
    );
    data.add_prism_face(
        [t00, t10, t11, t01],
        boundary.z().into(),
        FaceStyle::FillAndOutline,
    );
    data.add_prism_face(
        [b10, b11, t11, t10],
        boundary.x().into(),
        FaceStyle::FillAndOutline,
    );
    data.add_prism_face(
        [b00, t00, t01, b01],
        boundary.x().into(),
        FaceStyle::FillAndOutline,
    );
    data.add_prism_face(
        [b01, t01, t11, b11],
        boundary.y().into(),
        FaceStyle::FillAndOutline,
    );
    data.add_prism_face(
        [b00, b10, t10, t00],
        boundary.y().into(),
        FaceStyle::FillAndOutline,
    );
    data
}

fn patch_rotation_as_gltf_data(kind: PatchRotationKind, pipe_length: f32) -> GltfData {
    let mut data = GltfData::default();
    let stride = pipe_length + 1.0;
    let movement = kind.movement_3d();
    let colors = PatchRotationColors::new(kind);

    let (min, max) = patch_rotation_bounds(movement, stride);

    add_patch_rotation_boundary_faces(&mut data, kind, stride, colors);

    data.add_prism_face(
        [
            vec3(min.x, min.y, min.z),
            vec3(min.x, max.y, min.z),
            vec3(max.x, max.y, min.z),
            vec3(max.x, min.y, min.z),
        ],
        colors.start_time,
        FaceStyle::FillAndOutline,
    );
    data.add_prism_face(
        [
            vec3(min.x, min.y, max.z),
            vec3(max.x, min.y, max.z),
            vec3(max.x, max.y, max.z),
            vec3(min.x, max.y, max.z),
        ],
        colors.end_time,
        FaceStyle::FillAndOutline,
    );
    data
}

#[derive(Debug, Clone, Copy)]
struct PatchRotationColors {
    before: RGBA,
    after: RGBA,
    start_time: RGBA,
    end_time: RGBA,
    mixed_side: PatchRotationMixedSide,
}

impl PatchRotationColors {
    fn new(kind: PatchRotationKind) -> Self {
        let before = RGBA::from(kind.x_axis_boundary_basis());
        let after = RGBA::from(kind.x_axis_boundary_basis().flip());
        let mixed_side = PatchRotationMixedSide::for_kind(kind);
        Self {
            before,
            after,
            start_time: after,
            end_time: before,
            mixed_side,
        }
    }
}

fn add_patch_rotation_boundary_faces(
    data: &mut GltfData,
    kind: PatchRotationKind,
    stride: f32,
    colors: PatchRotationColors,
) {
    let min_u = -0.5;
    let max_u = 0.5;
    let min_v = -0.5;
    let max_v = stride + 0.5;
    let min_z = -0.5;
    let max_z = stride + 0.5;
    let mid_v = f32::midpoint(min_v, max_v);
    let mid_z = f32::midpoint(min_z, max_z);

    add_patch_rotation_local_face(
        data,
        kind,
        [
            vec3(min_u, min_v, min_z),
            vec3(max_u, min_v, min_z),
            vec3(max_u, min_v, max_z),
            vec3(min_u, min_v, max_z),
        ],
        colors.after,
    );
    add_patch_rotation_local_face(
        data,
        kind,
        [
            vec3(min_u, max_v, min_z),
            vec3(min_u, max_v, max_z),
            vec3(max_u, max_v, max_z),
            vec3(max_u, max_v, min_z),
        ],
        colors.before,
    );

    for (side, u) in [
        (PatchRotationMixedSide::Left, min_u),
        (PatchRotationMixedSide::Right, max_u),
    ] {
        if side == colors.mixed_side {
            add_patch_rotation_local_face(
                data,
                kind,
                [
                    vec3(u, min_v, min_z),
                    vec3(u, mid_v, min_z),
                    vec3(u, mid_v, max_z),
                    vec3(u, min_v, max_z),
                ],
                colors.before,
            );
            add_patch_rotation_local_face(
                data,
                kind,
                [
                    vec3(u, mid_v, min_z),
                    vec3(u, max_v, min_z),
                    vec3(u, max_v, max_z),
                    vec3(u, mid_v, max_z),
                ],
                colors.after,
            );
        } else {
            add_patch_rotation_local_face(
                data,
                kind,
                [
                    vec3(u, min_v, min_z),
                    vec3(u, max_v, min_z),
                    vec3(u, max_v, mid_z),
                    vec3(u, min_v, mid_z),
                ],
                colors.before,
            );
            add_patch_rotation_local_face(
                data,
                kind,
                [
                    vec3(u, min_v, mid_z),
                    vec3(u, max_v, mid_z),
                    vec3(u, max_v, max_z),
                    vec3(u, min_v, max_z),
                ],
                colors.after,
            );
        }
    }
}

fn add_patch_rotation_local_face(
    data: &mut GltfData,
    kind: PatchRotationKind,
    mut corners: [Vec3; 4],
    color: RGBA,
) {
    // The left face and the +X coordinate reflection each reverse orientation.
    if corners.iter().all(|corner| corner.x < 0.0) ^ (kind.movement().x == 1) {
        corners.reverse();
    }
    data.add_prism_face(
        corners.map(|corner| patch_rotation_local_to_global(kind, corner)),
        color,
        FaceStyle::FillAndOutline,
    );
}

fn patch_rotation_local_to_global(kind: PatchRotationKind, point: Vec3) -> Vec3 {
    match kind.movement().to_array() {
        [0, 1] => point,
        [0, -1] => vec3(-point.x, -point.y, point.z),
        [1, 0] => vec3(point.y, point.x, point.z),
        [-1, 0] => vec3(-point.y, point.x, point.z),
        _ => unreachable!("PatchRotationKind rejects non-cardinal movement"),
    }
}

fn patch_rotation_local_vector_to_global(kind: PatchRotationKind, vector: Vec3) -> Vec3 {
    patch_rotation_local_to_global(kind, vector) - patch_rotation_local_to_global(kind, Vec3::ZERO)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatchRotationMixedSide {
    Left,
    Right,
}

impl PatchRotationMixedSide {
    fn for_kind(kind: PatchRotationKind) -> Self {
        match (
            kind.x_axis_boundary_basis(),
            kind.movement() == glam::IVec2::NEG_X,
        ) {
            (Basis::X, true) | (Basis::Z, false) => Self::Right,
            (Basis::X, false) | (Basis::Z, true) => Self::Left,
        }
    }
}

fn patch_rotation_bounds(movement: IVec3, stride: f32) -> (Vec3, Vec3) {
    let min = vec3(
        (0.min(movement.x) as f32) * stride - 0.5,
        (0.min(movement.y) as f32) * stride - 0.5,
        -0.5,
    );
    let max = vec3(
        (0.max(movement.x) as f32) * stride + 0.5,
        (0.max(movement.y) as f32) * stride + 0.5,
        stride + 0.5,
    );
    (min, max)
}

fn has_pipe_toward(graph: &BlockGraph, position: IVec3, offset: IVec3) -> bool {
    crate::checked_add_position(position, offset)
        .is_ok_and(|neighbor| graph.has_pipe_between(position, neighbor))
}

fn checked_position_delta(lhs: IVec3, rhs: IVec3) -> Option<IVec3> {
    Some(IVec3::new(
        lhs.x.checked_sub(rhs.x)?,
        lhs.y.checked_sub(rhs.y)?,
        lhs.z.checked_sub(rhs.z)?,
    ))
}

/// Builds the geometry for a single block, with unit pipe length.
pub fn block_as_gltf_data(block: &Block, g: &BlockGraph) -> GltfData {
    block_as_gltf_data_with_pipe_length(block, g, 1.0)
}

/// Builds the geometry for a single block, using `pipe_length` to size the gaps
/// between blocks that pipes span.
///
/// The graph `g` is consulted for neighbor context (e.g. boundary orientation).
///
/// # Panics
///
/// Panics if an internally constructed Port block lacks its display color.
pub fn block_as_gltf_data_with_pipe_length(
    block: &Block,
    g: &BlockGraph,
    pipe_length: f32,
) -> GltfData {
    let mut data = GltfData::default();
    let p = block.pos;
    let center = Vec3::ZERO;

    match block.kind {
        BlockKind::Port => {
            data.add_cube(
                center,
                Vec3::ONE,
                block
                    .port_color()
                    .expect("Port blocks have a display color"),
                FaceStyle::FillAndOutline,
            );
            let connected_faces = Direction::iter()
                .filter(|direction| has_pipe_toward(g, p, direction.to_ivec3()))
                .collect::<Vec<_>>();
            data = data.pop_faces_at_directions(&connected_faces);
        }
        BlockKind::Cube(kind) => {
            let height_cells = block.height_cells();
            if height_cells == 1 {
                for dir in Direction::iter() {
                    if !has_pipe_toward(g, p, dir.to_ivec3()) {
                        let d_vec = dir.to_vec3();
                        let basis = get_basis(dir.as_udirection(), kind);
                        let color = basis.into();

                        let u = dir.as_udirection();
                        let next_u = next_axis(u);
                        let next_next_u = next_axis(next_u);

                        let s1_u = match dir {
                            Direction::XPLUS | Direction::YPLUS | Direction::ZPLUS => next_u,
                            Direction::XMINUS | Direction::YMINUS | Direction::ZMINUS => {
                                next_next_u
                            }
                        };
                        let s2_u = match dir {
                            Direction::XPLUS | Direction::YPLUS | Direction::ZPLUS => next_next_u,
                            Direction::XMINUS | Direction::YMINUS | Direction::ZMINUS => next_u,
                        };

                        let s1 = s1_u.to_vec3();
                        let s2 = s2_u.to_vec3();

                        data.add_face(
                            center + d_vec * 0.5,
                            s1,
                            s2,
                            color,
                            FaceStyle::FillAndOutline,
                        );
                    }
                }
                data.add_cube(center, Vec3::ONE, RGBA::LINE_BLACK, FaceStyle::Outline);
            } else {
                add_tall_cube_faces(&mut data, g, p, kind, height_cells, pipe_length);
            }
        }
        BlockKind::Walking(kind) => {
            data.extend(walking_as_gltf_data(kind, pipe_length));
        }
        BlockKind::PatchRotation(kind) => {
            data.extend(patch_rotation_as_gltf_data(kind, pipe_length));
        }
        BlockKind::T => data.add_cube(center, Vec3::ONE, RGBA::T_PURPLE, FaceStyle::FillAndOutline),
        BlockKind::Y => {
            let center = if has_pipe_toward(g, p, IVec3::Z) {
                center + Vec3::Z * 0.25
            } else if has_pipe_toward(g, p, IVec3::NEG_Z) {
                center + Vec3::NEG_Z * 0.25
            } else {
                center
            };
            data.add_cube(
                center,
                vec3(1.0, 1.0, 0.5),
                RGBA::Y_GREEN,
                FaceStyle::FillAndOutline,
            );
        }
        BlockKind::Measurement(basis) => data.add_cube(
            center + Vec3::NEG_Z * 0.25,
            vec3(1.0, 1.0, 0.5),
            basis.into(),
            FaceStyle::FillAndOutline,
        ),
        BlockKind::Selective(kind) => {
            let rgba1 = kind.pauli_if_true().into();
            let rgba2 = kind.pauli_if_false().into();
            data.add_double_color_cube(
                center + Vec3::NEG_Z * 0.25,
                vec3(1.0, 1.0, 0.5),
                rgba1,
                rgba2,
            );
        }
    }
    data
}

fn add_tall_cube_faces(
    data: &mut GltfData,
    graph: &BlockGraph,
    pos: IVec3,
    kind: crate::CubeKind,
    height_cells: u32,
    pipe_length: f32,
) {
    let stride = pipe_length + 1.0;
    let min = vec3(-0.5, -0.5, -0.5);
    let max = vec3(0.5, 0.5, (height_cells - 1) as f32 * stride + 0.5);
    let top = crate::checked_add_position(pos, IVec3::new(0, 0, height_cells as i32 - 1)).ok();
    let x_color: RGBA = kind.x().into();
    let y_color: RGBA = kind.y().into();
    let z_color: RGBA = kind.z().into();

    if !has_pipe_toward(graph, pos, IVec3::NEG_X) {
        data.add_prism_face(
            [
                vec3(min.x, min.y, min.z),
                vec3(min.x, min.y, max.z),
                vec3(min.x, max.y, max.z),
                vec3(min.x, max.y, min.z),
            ],
            x_color,
            FaceStyle::FillAndOutline,
        );
    }
    if !has_pipe_toward(graph, pos, IVec3::X) {
        data.add_prism_face(
            [
                vec3(max.x, min.y, min.z),
                vec3(max.x, max.y, min.z),
                vec3(max.x, max.y, max.z),
                vec3(max.x, min.y, max.z),
            ],
            x_color,
            FaceStyle::FillAndOutline,
        );
    }
    if !has_pipe_toward(graph, pos, IVec3::NEG_Y) {
        data.add_prism_face(
            [
                vec3(min.x, min.y, min.z),
                vec3(max.x, min.y, min.z),
                vec3(max.x, min.y, max.z),
                vec3(min.x, min.y, max.z),
            ],
            y_color,
            FaceStyle::FillAndOutline,
        );
    }
    if !has_pipe_toward(graph, pos, IVec3::Y) {
        data.add_prism_face(
            [
                vec3(min.x, max.y, min.z),
                vec3(min.x, max.y, max.z),
                vec3(max.x, max.y, max.z),
                vec3(max.x, max.y, min.z),
            ],
            y_color,
            FaceStyle::FillAndOutline,
        );
    }
    if !has_pipe_toward(graph, pos, IVec3::NEG_Z) {
        data.add_prism_face(
            [
                vec3(min.x, min.y, min.z),
                vec3(min.x, max.y, min.z),
                vec3(max.x, max.y, min.z),
                vec3(max.x, min.y, min.z),
            ],
            z_color,
            FaceStyle::FillAndOutline,
        );
    }
    if top.is_none_or(|top| !has_pipe_toward(graph, top, IVec3::Z)) {
        data.add_prism_face(
            [
                vec3(min.x, min.y, max.z),
                vec3(max.x, min.y, max.z),
                vec3(max.x, max.y, max.z),
                vec3(min.x, max.y, max.z),
            ],
            z_color,
            FaceStyle::FillAndOutline,
        );
    }
}

/// Builds the geometry for a pipe between two blocks.
pub fn pipe_as_gltf_data(
    u: &Block,
    v: &Block,
    pipe: &Pipe,
    g: &BlockGraph,
    pipe_length: f32,
) -> GltfData {
    pipe_between_positions_as_gltf_data(u.pos, v.pos, pipe, g, pipe_length)
}

/// Builds the geometry for a pipe between two lattice positions.
pub fn pipe_between_positions_as_gltf_data(
    u_pos: IVec3,
    v_pos: IVec3,
    pipe: &Pipe,
    g: &BlockGraph,
    pipe_length: f32,
) -> GltfData {
    let mut data = GltfData::default();
    let diff = v_pos.as_vec3() - u_pos.as_vec3();
    let pipe_height = scaled_spatial_pipe_height(u_pos, v_pos, pipe, g, pipe_length).unwrap_or(1.0);
    let center = Vec3::Z * ((pipe_height - 1.0) * 0.5);

    let h2 = HADAMARD_STRIP_WIDTH * pipe_length * 0.25;
    let p2 = pipe_length * 0.5;
    let face_center_delta = f32::midpoint(p2, h2);

    let main_axis = if diff.x != 0.0 {
        Vec3::X
    } else if diff.y != 0.0 {
        Vec3::Y
    } else {
        Vec3::Z
    };

    let inferred_bases = g.infer_pipe_basis(pipe);
    let pipe_dir_u = pipe.dir.as_udirection();

    let size = Vec3::ONE + main_axis.abs() * (pipe_length - 1.0) + Vec3::Z * (pipe_height - 1.0);
    data.add_cube(center, size, RGBA::LINE_BLACK, FaceStyle::Outline);

    for dir in UDirection::iter() {
        if dir == pipe_dir_u {
            continue;
        }

        // No basis is inferable for this face when the pipe attaches to a block
        // that defines no boundary color (Port/T/…). Render it as undefined gray
        // instead of silently masquerading as an X boundary.
        let (color_start, color_end, strip_color) = match inferred_bases[dir.index()] {
            Some(basis) => {
                let basis_end = if pipe.hadamard { basis.flip() } else { basis };
                let strip = if pipe.hadamard {
                    RGBA::H_YELLOW
                } else {
                    basis.into()
                };
                (basis.into(), basis_end.into(), strip)
            }
            None => (RGBA::PORT_GRAY, RGBA::PORT_GRAY, RGBA::PORT_GRAY),
        };

        let normal = dir.to_vec3();
        let normal_len = if normal.abs().dot(Vec3::Z) > 0.0 {
            pipe_height
        } else {
            1.0
        };
        let width_dir = main_axis.cross(normal);
        let width_len = if width_dir.abs().dot(Vec3::Z) > 0.0 {
            pipe_height
        } else {
            1.0
        };
        let width_dir = width_dir * width_len;

        for polarity in [1.0, -1.0] {
            let d = normal * polarity;
            let normal_offset = d * (normal_len * 0.5);

            let (s1, s2) = if main_axis.cross(width_dir).dot(d) < 0.0 {
                (width_dir, main_axis * (p2 - h2))
            } else {
                (main_axis * (p2 - h2), width_dir)
            };

            let (s1_strip, s2_strip) = if main_axis.cross(width_dir).dot(d) < 0.0 {
                (width_dir, main_axis * h2 * 2.0)
            } else {
                (main_axis * h2 * 2.0, width_dir)
            };

            data.add_face(
                center - diff * face_center_delta + normal_offset,
                s1,
                s2,
                color_start,
                FaceStyle::Fill,
            );
            data.add_face(
                center + diff * face_center_delta + normal_offset,
                s1,
                s2,
                color_end,
                FaceStyle::Fill,
            );
            data.add_face(
                center + normal_offset,
                s1_strip,
                s2_strip,
                strip_color,
                FaceStyle::Fill,
            );
        }
    }
    data
}

fn scaled_spatial_pipe_height(
    u_pos: IVec3,
    v_pos: IVec3,
    pipe: &Pipe,
    g: &BlockGraph,
    pipe_length: f32,
) -> Option<f32> {
    if !pipe.dir.is_spatial() {
        return None;
    }
    let u = g.get_endpoint_block(u_pos)?;
    let v = g.get_endpoint_block(v_pos)?;
    if !u.kind().is_cube() || !v.kind().is_cube() || u.height_cells() != v.height_cells() {
        return None;
    }
    (u.height_cells() > 1).then_some((u.height_cells() - 1) as f32 * (pipe_length + 1.0) + 1.0)
}

/// Builds the geometry for an entire block graph, laying blocks out on a grid
/// with `pipe_length`-sized gaps between adjacent cells.
pub fn block_graph_as_gltf_data(g: &BlockGraph, pipe_length: f32) -> GltfData {
    block_graph_as_gltf_data_with_popped_faces(g, pipe_length, &[])
}

/// Builds an entire block graph while removing the selected filled faces.
pub fn block_graph_as_gltf_data_with_popped_faces(
    g: &BlockGraph,
    pipe_length: f32,
    popped_faces: &[GltfFaceSelector],
) -> GltfData {
    block_graph_as_gltf_data_with_modules(g, pipe_length, popped_faces, None)
}

pub(crate) fn block_graph_as_gltf_data_with_modules(
    g: &BlockGraph,
    pipe_length: f32,
    popped_faces: &[GltfFaceSelector],
    modules: Option<&crate::ModuleView>,
) -> GltfData {
    build_module_geometry(g, pipe_length, popped_faces, modules, true).0
}

pub(crate) fn module_html_geometry(
    g: &BlockGraph,
    pipe_length: f32,
    popped_faces: &[GltfFaceSelector],
    modules: Option<&crate::ModuleView>,
) -> (GltfData, ModuleFaceRanges) {
    build_module_geometry(g, pipe_length, popped_faces, modules, false)
}

fn build_module_geometry(
    g: &BlockGraph,
    pipe_length: f32,
    popped_faces: &[GltfFaceSelector],
    modules: Option<&crate::ModuleView>,
    highlight: bool,
) -> (GltfData, ModuleFaceRanges) {
    let mut data = GltfData::default();
    let mut ranges = ModuleFaceRanges::new();
    let mut append = |mut object: GltfData, owner: Option<usize>, center: Vec3| {
        if highlight {
            if let Some(owner) = owner {
                object.color_module_faces(crate::module_color(owner));
            }
        } else {
            for (color, triangles) in &object.triangles {
                let start = data.triangles.get(color).map_or(0, Vec::len);
                let end = start + triangles.len();
                let spans = ranges.entry(*color).or_default();
                if let Some((last_owner, last_range)) = spans.last_mut()
                    && *last_owner == owner
                {
                    last_range.end = end;
                } else {
                    spans.push((owner, start..end));
                }
            }
        }
        data.extend(object.map_points(|v| v + center));
    };
    let stride = pipe_length + 1.0;

    for block in g.blocks() {
        let p = block.pos;
        let center = p.as_vec3() * stride;
        let directions = popped_faces
            .iter()
            .filter_map(|selector| selector.for_block(p))
            .collect::<Vec<_>>();
        let block_data = block_as_gltf_data_with_pipe_length(block, g, pipe_length)
            .pop_faces_at_directions(&directions);
        append(
            block_data,
            modules.and_then(|view| view.module_for_block(p)),
            center,
        );
    }

    for (u_pos, v_pos, _, _, pipe) in g.pipe_endpoints_with_blocks() {
        let diff = (v_pos - u_pos).as_vec3();
        let center = u_pos.as_vec3() * stride + diff * stride * 0.5;
        let directions = popped_faces
            .iter()
            .filter_map(|selector| selector.for_pipe(u_pos, v_pos))
            .collect::<Vec<_>>();
        let pipe_data = pipe_between_positions_as_gltf_data(u_pos, v_pos, pipe, g, pipe_length)
            .pop_faces_at_directions(&directions);
        append(
            pipe_data,
            modules.and_then(|view| view.module_for_pipe(g, u_pos, v_pos)),
            center,
        );
    }
    (data, ranges)
}

/// Builds the geometry visualizing a stabilizer's support overlaid on the graph.
///
/// # Errors
///
/// Returns a [`BlockGraphError`] if the stabilizer cannot be resolved against
/// the graph's derived faces.
pub fn stabilizer_as_gltf_data(
    generator: &StabilizerGenerator,
    g: &BlockGraph,
    pipe_length: f32,
) -> Result<GltfData, BlockGraphError> {
    let measurement_label = if let StabilizerRowKind::Measurement { name } = &generator.kind {
        ZXGraph::try_from(g)?
            .measurement_outcome_position(name)
            .map(|position| (position, name.clone()))
    } else {
        None
    };
    let stabilizer = &generator.stabilizer;
    let g = g.fix_shadowed_faces();
    let mut data = GltfData::default();
    let stride = pipe_length + 1.0;

    generate_interior_edge_surfaces(stabilizer, &g, pipe_length, stride, &mut data)?;
    let neighbors = build_neighbor_map(stabilizer, &g);
    generate_interior_node_surfaces(stabilizer, &g, stride, &neighbors, &mut data)?;

    if let Some((pos, text)) = measurement_label {
        data.texts.push(TextLabel {
            position: (pos + 0.25 * Vec3::Z) * stride,
            text,
            color: RGBA::H_YELLOW,
        });
    }
    Ok(data)
}

fn get_surface_normal(g: &BlockGraph, pipe: &Pipe, p: Pauli) -> Vec3 {
    let inferred = g.infer_pipe_basis(pipe);
    let pipe_dir = pipe.dir.as_udirection();

    UDirection::iter()
        .filter(|&d| d != pipe_dir)
        .find_map(|dir| {
            inferred[dir.index()].and_then(|basis| {
                if Pauli::from(basis) == p.flip() {
                    Some(dir.to_vec3())
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| pipe_dir.to_vec3())
}

fn generate_interior_edge_surfaces(
    stabilizer: &Stabilizer,
    g: &BlockGraph,
    pipe_length: f32,
    stride: f32,
    data: &mut GltfData,
) -> Result<(), BlockGraphError> {
    for (&(u, v), &p) in &stabilizer.interior_edges {
        let physical = physical_pipe_for_stabilizer_edge(g, u, v)
            .ok_or(BlockGraphError::PipeNotFound(u, v))?;
        let pipe = physical.pipe;

        let (u, v, p) = if physical.src == pipe.src {
            (physical.src, physical.dst, p)
        } else {
            (
                physical.dst,
                physical.src,
                if pipe.hadamard { p.flip() } else { p },
            )
        };
        let dir_vec = (v - u).as_vec3();
        let center = (u.as_vec3() + v.as_vec3()) * 0.5 * stride;

        let segment_len = if pipe.hadamard {
            pipe_length * 0.5
        } else {
            pipe_length
        };
        let pipe_height = scaled_spatial_pipe_height(u, v, pipe, g, pipe_length).unwrap_or(1.0);
        let scaled_pipe_center_offset = Vec3::Z * ((pipe_height - 1.0) * 0.5);

        for pauli in p.iter_xz() {
            let normal = get_surface_normal(g, pipe, pauli);
            let offset = if pipe.hadamard {
                -0.5 * segment_len * dir_vec
            } else {
                Vec3::ZERO
            };
            let seg_center = center + offset + scaled_pipe_center_offset;
            let width_vec = scale_z_dimension(dir_vec.cross(normal), pipe_height);

            data.add_face(
                seg_center,
                dir_vec * segment_len,
                width_vec,
                correlation_surface_color(pauli),
                FaceStyle::Fill,
            );

            if pipe.hadamard {
                let p_flipped = pauli.flip();
                let color_flipped = correlation_surface_color(p_flipped);
                let seg_center = center - offset + scaled_pipe_center_offset;
                data.add_face(
                    seg_center,
                    dir_vec * segment_len,
                    width_vec,
                    color_flipped,
                    FaceStyle::Fill,
                );
            }
        }
    }
    Ok(())
}

fn add_walking_node_surface(
    block_pos: IVec3,
    kind: WalkingKind,
    p: Pauli,
    stride: f32,
    data: &mut GltfData,
) {
    let start = walking_attachment_point(block_pos, kind, block_pos, stride);
    let end = walking_attachment_point(block_pos, kind, kind.end_position(block_pos), stride);
    add_walking_surface_between(kind, p, start, end, data);
}

fn add_walking_surface_between(
    kind: WalkingKind,
    p: Pauli,
    start: Vec3,
    end: Vec3,
    data: &mut GltfData,
) {
    let center = (start + end) * 0.5;
    let side = end - start;

    for pauli in p.iter_xz() {
        let Some(width_vec) = walking_surface_width(kind, pauli) else {
            continue;
        };
        data.add_face(
            center,
            side,
            width_vec,
            correlation_surface_color(pauli),
            FaceStyle::Fill,
        );
    }
}

fn walking_attachment_point(
    block_pos: IVec3,
    kind: WalkingKind,
    endpoint: IVec3,
    stride: f32,
) -> Vec3 {
    if endpoint == block_pos {
        endpoint.as_vec3() * stride + Vec3::NEG_Z * 0.5
    } else if endpoint == kind.end_position(block_pos) {
        endpoint.as_vec3() * stride + Vec3::Z * 0.5
    } else {
        endpoint.as_vec3() * stride
    }
}

fn walking_surface_width(kind: WalkingKind, pauli: Pauli) -> Option<Vec3> {
    let boundary = kind.boundary();
    [(boundary.x(), Vec3::X), (boundary.y(), Vec3::Y)]
        .into_iter()
        .find_map(|(basis, axis)| (Pauli::from(basis) == pauli).then_some(axis))
}

#[derive(Debug, Clone, Copy)]
struct PatchRotationStabilizerQuad {
    center: Vec3,
    side1: Vec3,
    side2: Vec3,
}

fn add_patch_rotation_node_surface(
    block_pos: IVec3,
    kind: PatchRotationKind,
    pauli: Pauli,
    stride: f32,
    data: &mut GltfData,
) {
    let Some(shape) = patch_rotation_stabilizer_shape(kind, pauli) else {
        return;
    };
    let color = correlation_surface_color(pauli);
    let translation = block_pos.as_vec3() * stride;

    for quad in
        patch_rotation_stabilizer_quads(shape, stride, PatchRotationMixedSide::for_kind(kind))
    {
        let center = patch_rotation_local_to_global(kind, quad.center) + translation;
        let side1 = patch_rotation_local_vector_to_global(kind, quad.side1);
        let side2 = patch_rotation_local_vector_to_global(kind, quad.side2);
        data.add_face(center, side1, side2, color, FaceStyle::Fill);
    }
}

fn patch_rotation_stabilizer_shape(kind: PatchRotationKind, pauli: Pauli) -> Option<Pauli> {
    let basis = Pauli::from(kind.x_axis_boundary_basis());
    if pauli == basis {
        Some(Pauli::X)
    } else if pauli == basis.flip() {
        Some(Pauli::Z)
    } else {
        None
    }
}

fn patch_rotation_stabilizer_quads(
    shape: Pauli,
    stride: f32,
    mixed_side: PatchRotationMixedSide,
) -> Vec<PatchRotationStabilizerQuad> {
    let min_x = -0.5;
    let max_x = 0.5;
    let mid_x = 0.0;
    let min_y = -0.5;
    let max_y = stride + 0.5;
    let min_z = -0.5;
    let max_z = stride + 0.5;
    let mid_y = f32::midpoint(min_y, max_y);
    let upper_line_y = f32::midpoint(min_y, mid_y);
    let upper_input_line_y = upper_line_y - 0.5;
    let lower_line_y = min_y + (max_y - min_y) * 0.75;
    let sheet_margin_y = (max_y - min_y) * 0.125;
    let upper_sheet_y = (upper_line_y - sheet_margin_y).max(min_y);
    let lower_sheet_y = (lower_line_y + sheet_margin_y).min(max_y);
    let lower_output_line_y = f32::midpoint(lower_line_y, max_y);
    let in_quarter_z = min_z + (max_z - min_z) * 0.25;
    let out_quarter_z = min_z + (max_z - min_z) * 0.75;
    let (mixed_x0, mixed_x1, extended_x0, extended_x1) = match mixed_side {
        PatchRotationMixedSide::Left => (min_x, mid_x, mid_x, max_x),
        PatchRotationMixedSide::Right => (mid_x, max_x, min_x, mid_x),
    };

    match shape {
        Pauli::Z => vec![
            xz_quad(mixed_x0, mixed_x1, mid_y, min_z, out_quarter_z),
            yz_quad(mid_x, min_y, mid_y, min_z, out_quarter_z),
            xy_quad(extended_x0, extended_x1, min_y, mid_y, out_quarter_z),
            xy_quad(min_x, max_x, mid_y, lower_sheet_y, out_quarter_z),
            xz_quad(min_x, max_x, lower_output_line_y, out_quarter_z, max_z),
        ],
        Pauli::X => vec![
            xz_quad(min_x, max_x, upper_input_line_y, min_z, in_quarter_z),
            xy_quad(min_x, max_x, upper_sheet_y, mid_y, in_quarter_z),
            xy_quad(extended_x0, extended_x1, mid_y, max_y, in_quarter_z),
            xz_quad(mixed_x0, mixed_x1, mid_y, in_quarter_z, max_z),
            yz_quad(mid_x, mid_y, max_y, in_quarter_z, max_z),
        ],
        _ => Vec::new(),
    }
}

fn xy_quad(x0: f32, x1: f32, y0: f32, y1: f32, z: f32) -> PatchRotationStabilizerQuad {
    PatchRotationStabilizerQuad {
        center: vec3(f32::midpoint(x0, x1), f32::midpoint(y0, y1), z),
        side1: Vec3::X * (x1 - x0),
        side2: Vec3::Y * (y1 - y0),
    }
}

fn xz_quad(x0: f32, x1: f32, y: f32, z0: f32, z1: f32) -> PatchRotationStabilizerQuad {
    PatchRotationStabilizerQuad {
        center: vec3(f32::midpoint(x0, x1), y, f32::midpoint(z0, z1)),
        side1: Vec3::X * (x1 - x0),
        side2: Vec3::Z * (z1 - z0),
    }
}

fn yz_quad(x: f32, y0: f32, y1: f32, z0: f32, z1: f32) -> PatchRotationStabilizerQuad {
    PatchRotationStabilizerQuad {
        center: vec3(x, f32::midpoint(y0, y1), f32::midpoint(z0, z1)),
        side1: Vec3::Y * (y1 - y0),
        side2: Vec3::Z * (z1 - z0),
    }
}

fn cube_stabilizer_height(block: &Block, stride: f32) -> f32 {
    if block.kind().is_cube() {
        (block.height_cells() - 1) as f32 * stride + 1.0
    } else {
        1.0
    }
}

fn cube_stabilizer_center(block: &Block, stride: f32) -> Vec3 {
    let height = cube_stabilizer_height(block, stride);
    block.pos().as_vec3() * stride + Vec3::Z * ((height - 1.0) * 0.5)
}

fn scale_z_dimension(vector: Vec3, height: f32) -> Vec3 {
    if vector.abs().dot(Vec3::Z) > 0.0 {
        vector * height
    } else {
        vector
    }
}

fn cube_half_surface_offset(dvec: Vec3, height: f32) -> f32 {
    if dvec.abs().dot(Vec3::Z) > 0.0 {
        height * 0.25
    } else {
        0.25
    }
}

fn cube_half_surface_side(dvec: Vec3, height: f32) -> Vec3 {
    if dvec.abs().dot(Vec3::Z) > 0.0 {
        dvec * (height * 0.5)
    } else {
        dvec * 0.5
    }
}

fn pipe_for_node_direction<'a>(
    g: &'a BlockGraph,
    block: &Block,
    dir: Direction,
) -> Option<(IVec3, &'a Pipe)> {
    block
        .connectable_offsets()
        .into_iter()
        .filter_map(|offset| crate::checked_add_position(block.pos(), offset).ok())
        .find_map(|endpoint| {
            crate::checked_add_position(endpoint, dir.to_ivec3())
                .ok()
                .and_then(|neighbor| g.get_pipe(endpoint, neighbor))
                .map(|pipe| (endpoint, pipe))
        })
}

fn build_neighbor_map(
    stabilizer: &Stabilizer,
    g: &BlockGraph,
) -> HashMap<IVec3, Vec<(Direction, Pauli)>> {
    let mut neighbors: HashMap<IVec3, Vec<(Direction, Pauli)>> =
        HashMap::with_capacity(stabilizer.interior_nodes.len());

    for (&(u, v), &p) in &stabilizer.interior_edges {
        if let Some(dir_uv) =
            checked_position_delta(v, u).and_then(|delta| Direction::try_from(delta).ok())
        {
            neighbors.entry(u).or_default().push((dir_uv, p));
            let p_v = g
                .get_pipe(u, v)
                .map(|pipe| if pipe.hadamard { p.flip() } else { p })
                .unwrap_or(p);
            if let Some(dir_vu) =
                checked_position_delta(u, v).and_then(|delta| Direction::try_from(delta).ok())
            {
                neighbors.entry(v).or_default().push((dir_vu, p_v));
            }
            continue;
        }

        let Some(physical) = physical_pipe_for_stabilizer_edge(g, u, v) else {
            continue;
        };

        let p_v = if physical.pipe.hadamard { p.flip() } else { p };
        if let Some(dir_uv) = checked_position_delta(physical.dst, physical.src)
            .and_then(|delta| Direction::try_from(delta).ok())
        {
            neighbors.entry(u).or_default().push((dir_uv, p));
            neighbors.entry(v).or_default().push((dir_uv.negate(), p_v));
        }
    }
    neighbors
}

struct PhysicalStabilizerPipe<'a> {
    src: IVec3,
    dst: IVec3,
    pipe: &'a Pipe,
}

fn physical_pipe_for_stabilizer_edge<'a>(
    g: &'a BlockGraph,
    u: IVec3,
    v: IVec3,
) -> Option<PhysicalStabilizerPipe<'a>> {
    let u_block = g.get_endpoint_block(u)?;
    let v_block = g.get_endpoint_block(v)?;
    for u_endpoint in u_block
        .connectable_offsets()
        .into_iter()
        .filter_map(|offset| crate::checked_add_position(u_block.pos(), offset).ok())
    {
        for v_endpoint in v_block
            .connectable_offsets()
            .into_iter()
            .filter_map(|offset| crate::checked_add_position(v_block.pos(), offset).ok())
        {
            let Some(pipe) = g.get_pipe(u_endpoint, v_endpoint) else {
                continue;
            };
            return Some(PhysicalStabilizerPipe {
                src: u_endpoint,
                dst: v_endpoint,
                pipe,
            });
        }
    }
    None
}

fn generate_interior_node_surfaces(
    stabilizer: &Stabilizer,
    g: &BlockGraph,
    stride: f32,
    neighbors: &HashMap<IVec3, Vec<(Direction, Pauli)>>,
    data: &mut GltfData,
) -> Result<(), BlockGraphError> {
    // A product can cancel a node's algebraic support while leaving a
    // through-going sheet on its incident pipes. Reconstruct cube coverage
    // from those pipes rather than dropping the middle of that sheet.
    let mut nodes = stabilizer.interior_nodes.clone();
    for (&pos, incident) in neighbors {
        if g.get_block(pos).is_some_and(|block| block.kind().is_cube()) {
            let support = incident.iter().fold(Pauli::I, |p, (_, q)| p | *q);
            let p = nodes.entry(pos).or_insert(Pauli::I);
            *p = *p | support;
        }
    }
    for (&pos, &p) in &nodes {
        let block = g
            .get_block(pos)
            .ok_or(BlockGraphError::BlockNotFound(pos))?;
        let center = pos.as_vec3() * stride;

        for pauli in p.iter_xz() {
            match block.kind {
                BlockKind::Walking(kind) => {
                    add_walking_node_surface(pos, kind, pauli, stride, data);
                }
                BlockKind::PatchRotation(kind) => {
                    add_patch_rotation_node_surface(pos, kind, pauli, stride, data);
                }
                BlockKind::Cube(kind) if Pauli::from(kind.normal_basis()) != pauli => {
                    let normal = kind.normal_direction().to_vec3();
                    if let Some(dvec) = Direction::iter()
                        .find(|d| d.as_udirection() != kind.normal_direction())
                        .map(Direction::to_vec3)
                    {
                        let center = cube_stabilizer_center(block, stride);
                        let width_vec = scale_z_dimension(
                            dvec.cross(normal),
                            cube_stabilizer_height(block, stride),
                        );
                        data.add_face(
                            center,
                            dvec,
                            width_vec,
                            correlation_surface_color(pauli),
                            FaceStyle::Fill,
                        );
                    }
                }
                BlockKind::Cube(_) | BlockKind::Port => {
                    let is_cube = block.kind.is_cube();
                    let height = cube_stabilizer_height(block, stride);
                    let center = if is_cube {
                        cube_stabilizer_center(block, stride)
                    } else {
                        center
                    };

                    for (dir, _) in neighbors
                        .get(&pos)
                        .into_iter()
                        .flatten()
                        .filter(|(_, neighbor_pauli)| *neighbor_pauli & pauli)
                    {
                        let dvec = dir.to_vec3();
                        if let Some((endpoint, pipe)) = pipe_for_node_direction(g, block, *dir) {
                            // `get_surface_normal` reads the pipe basis from its
                            // source endpoint, so translate the local pauli into
                            // that frame (a Hadamard flips X<->Z across the pipe)
                            // when this node sits on the pipe's dst side. The face
                            // color still uses the un-flipped local `pauli`, which
                            // is this node's own correlation surface.
                            let effective_pauli = if pipe.hadamard && (endpoint != pipe.src) {
                                pauli.flip()
                            } else {
                                pauli
                            };
                            let normal = get_surface_normal(g, pipe, effective_pauli);
                            let (width_vec, side, offset) = if is_cube {
                                (
                                    scale_z_dimension(dvec.cross(normal), height),
                                    cube_half_surface_side(dvec, height),
                                    cube_half_surface_offset(dvec, height),
                                )
                            } else {
                                (dvec.cross(normal), dvec * 0.5, 0.25)
                            };

                            data.add_face(
                                center + dvec * offset,
                                side,
                                width_vec,
                                correlation_surface_color(pauli),
                                FaceStyle::Fill,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn correlation_surface_color(pauli: Pauli) -> RGBA {
    match pauli {
        Pauli::X => RGBA::X_PURE_RED,
        Pauli::Y => RGBA::Y_PURE_GREEN,
        Pauli::Z => RGBA::Z_PURE_BLUE,
        Pauli::I => RGBA::LINE_BLACK,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GalleryItem, WalkingBoundaryKind, parse_blog_to_graph};
    use bloq_utils::PauliString;
    use glam::{IVec2, ivec2};

    const EPS: f32 = 1e-5;

    #[test]
    fn gltf_bakes_editor_lighting_without_changing_palette_opacity() {
        use gltf_json::validation::Validate;

        let mut data = GltfData::default();
        // +Y and -Y faces exercise the diagonal key and ambient-only sides.
        data.add_triangles(
            RGBA::PORT_GRAY,
            [
                [Vec3::ZERO, Vec3::Z, Vec3::X],
                [Vec3::ZERO, Vec3::X, Vec3::Z],
            ],
        );
        data.add_line(RGBA::LINE_BLACK, [[Vec3::ZERO, Vec3::X]]);
        let model = data.to_model();
        let bytes = gltf_json::serialize::to_vec(&model).unwrap();
        let model: Root = gltf_json::deserialize::from_slice(&bytes).unwrap();
        let mut errors = Vec::new();
        model.validate(&model, gltf_json::Path::new, &mut |path, error| {
            errors.push((path(), error));
        });
        assert!(errors.is_empty(), "invalid glTF: {errors:?}");
        assert!(
            model
                .extensions_required
                .iter()
                .any(|name| name == "KHR_materials_unlit")
        );
        assert!(
            model
                .materials
                .iter()
                .all(|material| { material.extensions.as_ref().unwrap().unlit.is_some() })
        );

        let primitive = &model.meshes[0].primitives[0];
        let material = &model.materials[primitive.material.unwrap().value()];
        assert_eq!(material.alpha_mode, Valid(AlphaMode::Blend));
        let [red, _, _, alpha] = material.pbr_metallic_roughness.base_color_factor.0;
        assert!((red - 0.7230551).abs() < EPS); // sRGB #DD decoded to linear.
        assert!((alpha - 89.0 / 255.0).abs() < EPS);
        let positions = &model.accessors[primitive.attributes[&Valid(Semantic::Positions)].value()];
        let colors = &model.accessors[primitive.attributes[&Valid(Semantic::Colors(0))].value()];
        assert_eq!(colors.type_, Valid(Type::Vec4));
        assert_eq!(colors.count, positions.count);
        assert_eq!(colors.count.0, 6);
        assert!(
            !model.meshes[0].primitives[1]
                .attributes
                .contains_key(&Valid(Semantic::Colors(0)))
        );

        let uri = model.buffers[0].uri.as_ref().unwrap();
        let buffer = BASE64_STANDARD
            .decode(uri.split_once(',').unwrap().1)
            .unwrap();
        let view = &model.buffer_views[colors.buffer_view.unwrap().value()];
        let start = view.byte_offset.unwrap().0 as usize;
        let end = start + view.byte_length.0 as usize;
        assert_eq!(end, buffer.len());
        let values: Vec<_> = buffer[start..end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        let exposure = 2.0_f32.powf(-9.7) / 1.2;
        let ambient = 0.4524 * EDITOR_AMBIENT_BRIGHTNESS * exposure;
        let lit = ambient + 0.3255 * EDITOR_DIRECTIONAL_ILLUMINANCE * exposure / 3.0_f32.sqrt();
        for (vertex, expected) in values
            .as_chunks::<4>()
            .0
            .iter()
            .zip([lit, lit, lit, ambient, ambient, ambient])
        {
            assert!((vertex[0] - expected).abs() < EPS);
            assert_eq!(vertex[0], vertex[1]);
            assert_eq!(vertex[1], vertex[2]);
            assert_eq!(vertex[3], 1.0); // Opacity is applied only by the material.
        }
        assert!(linear_palette(RGBA::from_hex(0x080808, 255))[0] < 0.003);
    }

    #[test]
    fn module_faces_keep_the_shared_opaque_color_and_unlit_brightness() {
        let mut data = GltfData::default();
        data.add_triangles(RGBA::X_RED, [[Vec3::ZERO, Vec3::X, Vec3::Y]]);
        data.add_line(RGBA::LINE_BLACK, [[Vec3::ZERO, Vec3::X]]);
        let color = crate::module_color(0);
        data.color_module_faces(color);
        assert_eq!(data.triangles.keys().copied().collect::<Vec<_>>(), [color]);
        assert!(data.lines.contains_key(&RGBA::LINE_BLACK));
        let data = data.swap_yz();
        let model = data.to_model();
        let primitive = &model.meshes[0].primitives[0];
        let material = &model.materials[primitive.material.unwrap().value()];
        assert_eq!(material.alpha_mode, Valid(AlphaMode::Opaque));
        assert_eq!(
            material.pbr_metallic_roughness.base_color_factor.0,
            linear_palette(color)
        );
        let accessor = &model.accessors[primitive.attributes[&Valid(Semantic::Colors(0))].value()];
        let view = &model.buffer_views[accessor.buffer_view.unwrap().value()];
        let buffer = data.to_buffer_bytes();
        let start = view.byte_offset.unwrap().0 as usize;
        for bytes in buffer[start..].as_chunks::<4>().0 {
            assert_eq!(f32::from_le_bytes(*bytes), 1.0);
        }
    }

    #[test]
    fn module_toggles_preserve_original_faces_opacity_and_seams() {
        let source = crate::GalleryItem::ThreeBitAdder.build();
        let graph = source.flatten().unwrap();
        let modules = crate::ModuleView::from_graph(&source, &graph).unwrap();
        let (data, ranges) = module_html_geometry(&graph, 2.0, &[], Some(&modules));
        let ordinary = block_graph_as_gltf_data(&graph, 2.0);
        assert_eq!(data.triangles, ordinary.triangles);
        assert_eq!(data.lines, ordinary.lines);
        assert_eq!(data.to_buffer_bytes(), ordinary.to_buffer_bytes());

        let (model, toggles) = data.module_model(&ranges);
        assert_eq!(model.buffers[0].uri, ordinary.to_model().buffers[0].uri);
        let mut primitive_index = 0;
        let mut owners = BTreeSet::new();
        let mut seams = 0;
        let mut transparent = 0;
        for (color, triangles) in &data.triangles {
            let mut next_triangle = 0;
            for (owner, range) in &ranges[color] {
                assert_eq!(range.start, next_triangle);
                next_triangle = range.end;
                transparent += usize::from(color.a < 255);
                let primitive = &model.meshes[0].primitives[primitive_index];
                primitive_index += 1;
                let positions =
                    &model.accessors[primitive.attributes[&Valid(Semantic::Positions)].value()];
                let colors =
                    &model.accessors[primitive.attributes[&Valid(Semantic::Colors(0))].value()];
                assert_eq!(positions.byte_offset.unwrap().0 as usize, range.start * 36);
                assert_eq!(colors.byte_offset.unwrap().0 as usize, range.start * 48);
                assert_eq!(positions.count.0 as usize, range.len() * 3);
                assert_eq!(positions.count, colors.count);
                let index = primitive.material.unwrap().value();
                let material = &model.materials[index];
                if let Some(owner) = owner {
                    owners.insert(*owner);
                    let (_, _, original, highlight, alpha) = toggles
                        .iter()
                        .find(|(group, material, ..)| group == owner && *material == index)
                        .unwrap();
                    assert_eq!(*original, linear_palette(*color));
                    assert_eq!(*highlight, linear_palette(crate::module_color(*owner)));
                    assert_eq!(
                        material.pbr_metallic_roughness.base_color_factor.0,
                        *highlight
                    );
                    assert_eq!(*alpha, if color.a < 255 { "BLEND" } else { "OPAQUE" });
                } else {
                    seams += 1;
                    assert_eq!(
                        material.alpha_mode,
                        Valid(if color.a < 255 {
                            AlphaMode::Blend
                        } else {
                            AlphaMode::Opaque
                        })
                    );
                    assert_eq!(
                        material.pbr_metallic_roughness.base_color_factor.0,
                        linear_palette(*color)
                    );
                }
            }
            assert_eq!(next_triangle, triangles.len());
        }
        assert_eq!(owners.len(), modules.modules().len());
        assert!(seams > 0);
        assert!(transparent > 0);
        assert_eq!(
            model.meshes[0].primitives.len() - primitive_index,
            data.lines.len()
        );
    }

    #[test]
    fn html_viewer_uses_editor_projection_and_no_tone_mapping() {
        let html = GltfData::default().to_html_str().unwrap();
        assert!(html.contains("model-viewer/4.2.0/"));
        assert!(html.contains("background-color: #D8E0E6"));
        assert!(html.contains("camera-orbit=\"45deg 54.7356deg auto\""));
        assert!(html.contains("field-of-view=\"45deg\""));
        assert!(html.contains("max-field-of-view=\"45deg\""));
        assert!(html.contains("tone-mapping=\"none\""));
        assert!(html.contains("aria-label=\"Reset graph view\""));
        assert!(html.contains("Graph axes: X and Y are spatial, Z is time"));
        for axis in ["X", "Y", "Z"] {
            assert!(html.contains(&format!("data-axis=\"{axis}\"")));
        }
        assert!(html.contains("viewer.addEventListener('camera-change', updateAxes)"));
    }

    fn lerp(min: f32, max: f32, t: f32) -> f32 {
        min + (max - min) * t
    }

    #[test]
    fn product_surface_keeps_cube_between_supported_pipes() {
        let graph = crate::GalleryItem::CNOT
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let mut z_sheets = graph.stabilizers().unwrap().into_iter().filter(|row| {
            row.stabilizer
                .port_stabilizer
                .values()
                .all(|&pauli| matches!(pauli, Pauli::I | Pauli::Z))
        });
        let mut product = z_sheets.next().unwrap();
        product
            .stabilizer
            .phase_free_mul_assign(&z_sheets.next().unwrap().stabilizer);
        let data = stabilizer_as_gltf_data(&product, &graph, 2.0).unwrap();
        assert_patch_rotation_color_at_named(
            &data,
            Vec3::Z * 3.0,
            correlation_surface_color(Pauli::Z),
            "Z control sheet crosses its lower junction",
        );
    }

    #[test]
    fn moving_prisms_face_outward_and_pop_the_geometric_top() {
        let mut kinds = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                if x == 0 && y == 0 {
                    continue;
                }
                let movement = ivec2(x, y);
                kinds.push(BlockKind::Walking(
                    WalkingKind::new(WalkingBoundaryKind::ZXZ, movement).unwrap(),
                ));
                if x == 0 || y == 0 {
                    for basis in [Basis::X, Basis::Z] {
                        kinds.push(BlockKind::PatchRotation(
                            PatchRotationKind::new(basis, movement).unwrap(),
                        ));
                    }
                }
            }
        }
        for kind in kinds {
            let block = Block::new(IVec3::ZERO, kind);
            let data = block_as_gltf_data(&block, &BlockGraph::new());
            let (min, max) = triangle_bounds(&data);
            let center = (min + max) * 0.5;
            for &[a, b, c] in data.triangles.values().flatten() {
                assert!(
                    (b - a).cross(c - a).dot(a - center) > EPS,
                    "inward face: {kind:?} {a:?} {b:?} {c:?}"
                );
            }
            let popped = data.pop_faces_at_directions(&[Direction::ZPLUS]);
            let count_at = |z: f32| {
                popped
                    .triangles
                    .values()
                    .flatten()
                    .filter(|triangle| triangle.iter().all(|vertex| (vertex.z - z).abs() < EPS))
                    .count()
            };
            assert_eq!(count_at(max.z), 0, "top of {kind:?}");
            assert_eq!(count_at(min.z), 2, "bottom of {kind:?}");
        }
    }

    #[test]
    fn popping_face_direction_removes_only_that_filled_face() {
        let graph = GalleryItem::XMemory
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let data = block_graph_as_gltf_data(&graph, 2.0);
        let triangle_count = data.triangles.values().map(Vec::len).sum::<usize>();
        let line_count = data.lines.values().map(Vec::len).sum::<usize>();

        let popped = data.pop_faces_at_directions(&[Direction::YMINUS]);

        assert!(popped.triangles.values().map(Vec::len).sum::<usize>() < triangle_count);
        assert_eq!(
            popped.lines.values().map(Vec::len).sum::<usize>(),
            line_count
        );
        assert!(popped.triangles.values().flatten().all(|[a, b, c]| {
            let normal = (*b - *a).cross(*c - *a).normalize_or_zero();
            normal.dot(Direction::YMINUS.to_vec3()) < 0.999
        }));
    }

    #[test]
    fn popping_selected_faces_leaves_other_elements_closed() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Cube(crate::CubeKind::ZXZ),
        ));
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(crate::CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));

        let triangle_count = |data: &GltfData| data.triangles.values().map(Vec::len).sum::<usize>();
        let closed = block_graph_as_gltf_data(&graph, 2.0);
        let scoped = block_graph_as_gltf_data_with_popped_faces(
            &graph,
            2.0,
            &[
                GltfFaceSelector::Block {
                    position: IVec3::ZERO,
                    face: Direction::YMINUS,
                },
                GltfFaceSelector::Pipe {
                    u: IVec3::ZERO,
                    v: IVec3::X,
                    face: Direction::YMINUS,
                },
            ],
        );
        let global = block_graph_as_gltf_data_with_popped_faces(
            &graph,
            2.0,
            &[GltfFaceSelector::All(Direction::YMINUS)],
        );

        assert_eq!(triangle_count(&closed) - triangle_count(&scoped), 8);
        assert_eq!(triangle_count(&closed) - triangle_count(&global), 10);
    }

    #[test]
    fn connected_port_omits_its_pipe_face_but_keeps_the_outline() {
        let port = Block::new(IVec3::Z, BlockKind::Port);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::ZERO,
            BlockKind::Cube(crate::CubeKind::ZXZ),
        ));
        graph.add_block(port.clone());
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));

        let connected = block_as_gltf_data(&port, &graph);
        let mut isolated_graph = BlockGraph::new();
        isolated_graph.add_block(port.clone());
        let isolated = block_as_gltf_data(&port, &isolated_graph);

        assert_eq!(
            isolated.triangles.values().map(Vec::len).sum::<usize>()
                - connected.triangles.values().map(Vec::len).sum::<usize>(),
            2
        );
        assert_eq!(
            connected.lines.values().map(Vec::len).sum::<usize>(),
            isolated.lines.values().map(Vec::len).sum::<usize>()
        );
        assert!(connected.triangles.values().flatten().all(|[a, b, c]| {
            let normal = (*b - *a).cross(*c - *a).normalize_or_zero();
            normal.dot(Direction::ZMINUS.to_vec3()) < 0.999
        }));
    }

    #[test]
    fn custom_port_color_keeps_port_alpha() {
        let port = Block::new(IVec3::ZERO, BlockKind::Port)
            .with_port_color([0xeb, 0x40, 0x34])
            .expect("Port accepts a color");
        let data = block_as_gltf_data(&port, &BlockGraph::new());

        assert!(
            data.triangles
                .contains_key(&RGBA::from_hex(0xeb4034, RGBA::PORT_GRAY.a))
        );
    }

    fn point_in_triangle(point: Vec3, triangle: [Vec3; 3]) -> bool {
        let [a, b, c] = triangle;
        let normal = (b - a).cross(c - a);
        if normal.length_squared() < EPS {
            return false;
        }
        if (point - a).dot(normal).abs() > EPS {
            return false;
        }

        let ab = b - a;
        let ac = c - a;
        let offset = point - a;
        let dot_ab_ab = ab.dot(ab);
        let dot_ab_ac = ab.dot(ac);
        let dot_ab_offset = ab.dot(offset);
        let dot_ac_ac = ac.dot(ac);
        let dot_ac_offset = ac.dot(offset);
        let denom = dot_ab_ab * dot_ac_ac - dot_ab_ac * dot_ab_ac;
        if denom.abs() < EPS {
            return false;
        }
        let inv_denom = 1.0 / denom;
        let u = (dot_ac_ac * dot_ab_offset - dot_ab_ac * dot_ac_offset) * inv_denom;
        let v = (dot_ab_ab * dot_ac_offset - dot_ab_ac * dot_ab_offset) * inv_denom;

        u >= -EPS && v >= -EPS && u + v <= 1.0 + EPS
    }

    fn colors_at(data: &GltfData, point: Vec3) -> Vec<RGBA> {
        let mut colors: Vec<_> = data
            .triangles
            .iter()
            .filter_map(|(color, triangles)| {
                triangles
                    .iter()
                    .any(|&triangle| point_in_triangle(point, triangle))
                    .then_some(*color)
            })
            .collect();
        colors.sort();
        colors.dedup();
        colors
    }

    fn assert_patch_rotation_color_at(data: &GltfData, point: Vec3, expected: RGBA) {
        let colors = colors_at(data, point);
        assert_eq!(colors, vec![expected], "unexpected color at {point:?}");
    }

    fn assert_patch_rotation_color_at_named(
        data: &GltfData,
        point: Vec3,
        expected: RGBA,
        name: &str,
    ) {
        let colors = colors_at(data, point);
        assert_eq!(
            colors,
            vec![expected],
            "unexpected color for {name} at {point:?}"
        );
    }

    fn assert_patch_rotation_empty_at_named(data: &GltfData, point: Vec3, name: &str) {
        let colors = colors_at(data, point);
        assert!(
            colors.is_empty(),
            "expected no surface for {name} at {point:?}, got {colors:?}"
        );
    }

    fn triangle_bounds(data: &GltfData) -> (Vec3, Vec3) {
        let mut vertices = data.triangles.values().flatten().flatten().copied();
        let first = vertices.next().expect("mesh should contain triangles");
        vertices.fold((first, first), |(min, max), vertex| {
            (min.min(vertex), max.max(vertex))
        })
    }

    fn assert_patch_rotation_time_colors(
        data: &GltfData,
        min: Vec3,
        max: Vec3,
        z_min_color: RGBA,
        z_max_color: RGBA,
    ) {
        assert_patch_rotation_color_at(
            data,
            vec3(lerp(min.x, max.x, 0.37), lerp(min.y, max.y, 0.23), min.z),
            z_min_color,
        );
        assert_patch_rotation_color_at(
            data,
            vec3(lerp(min.x, max.x, 0.29), lerp(min.y, max.y, 0.61), max.z),
            z_max_color,
        );
    }

    fn assert_patch_rotation_colors(basis: Basis, movement: IVec2) {
        let pipe_length = 1.0;
        let stride = pipe_length + 1.0;
        let kind = PatchRotationKind::new(basis, movement).unwrap();
        let data = patch_rotation_as_gltf_data(kind, pipe_length);
        let (min, max) = patch_rotation_bounds(IVec3::new(movement.x, movement.y, 1), stride);
        let colors = PatchRotationColors::new(kind);
        let measurement_basis = kind.x_axis_boundary_basis();

        assert_patch_rotation_time_colors(&data, min, max, colors.start_time, colors.end_time);
        assert_eq!(colors.start_time, RGBA::from(measurement_basis.flip()));
        assert_eq!(colors.end_time, RGBA::from(measurement_basis));

        let local = |x: f32, y: f32, z: f32| patch_rotation_local_to_global(kind, vec3(x, y, z));
        let mixed_x = match colors.mixed_side {
            PatchRotationMixedSide::Left => -0.5,
            PatchRotationMixedSide::Right => 0.5,
        };
        let extended_x = -mixed_x;
        let mid_y = stride * 0.5;
        let low_z = 0.25;
        let high_z = stride - 0.25;

        assert_patch_rotation_color_at(&data, local(0.0, -0.5, low_z), colors.after);
        assert_patch_rotation_color_at(&data, local(0.0, stride + 0.5, low_z), colors.before);
        assert_patch_rotation_color_at(&data, local(mixed_x, mid_y * 0.5, low_z), colors.before);
        assert_patch_rotation_color_at(
            &data,
            local(mixed_x, mid_y + mid_y * 0.5, low_z),
            colors.after,
        );
        assert_patch_rotation_color_at(&data, local(extended_x, mid_y * 0.5, low_z), colors.before);
        assert_patch_rotation_color_at(&data, local(extended_x, mid_y * 0.5, high_z), colors.after);
    }

    #[test]
    fn boundary_blocks_render_with_overflowing_neighbor_directions_absent() {
        for position in [IVec3::splat(i32::MIN), IVec3::splat(i32::MAX)] {
            let block = Block::new(position, BlockKind::Cube(crate::CubeKind::ZXZ));
            let mut graph = BlockGraph::new();
            graph.try_add_block(block.clone()).unwrap();

            let data = block_as_gltf_data(&block, &graph);
            assert!(!data.triangles.is_empty());
        }

        let scaled = Block::new(
            IVec3::new(i32::MAX, 0, 0),
            BlockKind::Cube(crate::CubeKind::ZXZ),
        )
        .with_height("2d".parse().expect("valid height"))
        .unwrap();
        let mut scaled_graph = BlockGraph::new();
        scaled_graph.try_add_block(scaled.clone()).unwrap();
        assert!(
            !block_as_gltf_data(&scaled, &scaled_graph)
                .triangles
                .is_empty()
        );

        assert!(pipe_for_node_direction(&scaled_graph, &scaled, Direction::XPLUS).is_none());
    }

    #[test]
    fn boundary_y_block_uses_its_representable_temporal_neighbor() {
        let below = IVec3::new(0, 0, i32::MAX - 1);
        let position = IVec3::new(0, 0, i32::MAX);
        let block = Block::new(position, BlockKind::Y);
        let mut graph = BlockGraph::new();
        graph
            .try_add_block(Block::new(below, BlockKind::Port))
            .unwrap();
        graph.try_add_block(block.clone()).unwrap();
        graph
            .try_add_pipe(Pipe::new(below, Direction::ZPLUS))
            .unwrap();

        let data = block_as_gltf_data(&block, &graph);
        let (min, max) = triangle_bounds(&data);
        assert!((min.z + 0.5).abs() < EPS, "unexpected min z: {min:?}");
        assert!(max.z.abs() < EPS, "unexpected max z: {max:?}");
    }

    #[test]
    fn measurement_block_is_a_past_facing_basis_colored_cap() {
        for basis in [Basis::X, Basis::Z] {
            let block = Block::new(IVec3::ZERO, BlockKind::Measurement(basis));
            let mut graph = BlockGraph::new();
            graph.add_block(block.clone());

            let data = block_as_gltf_data(&block, &graph);
            let (min, max) = triangle_bounds(&data);
            assert!((min.z + 0.5).abs() < EPS, "unexpected min z: {min:?}");
            assert!(max.z.abs() < EPS, "unexpected max z: {max:?}");
            assert!(data.triangles.contains_key(&RGBA::from(basis)));
        }
    }

    #[test]
    fn patch_rotation_visualization_colors_all_movements() {
        for basis in [Basis::X, Basis::Z] {
            for movement in [ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1)] {
                assert_patch_rotation_colors(basis, movement);
            }
        }
    }

    #[test]
    fn patch_rotation_visualization_mixed_side_matches_construction_table() {
        use PatchRotationMixedSide::{Left, Right};

        let cases = [
            (Basis::X, ivec2(-1, 0), Left),
            (Basis::X, ivec2(1, 0), Right),
            (Basis::X, ivec2(0, -1), Left),
            (Basis::X, ivec2(0, 1), Left),
            (Basis::Z, ivec2(-1, 0), Right),
            (Basis::Z, ivec2(1, 0), Left),
            (Basis::Z, ivec2(0, -1), Right),
            (Basis::Z, ivec2(0, 1), Right),
        ];

        for (basis, movement, expected) in cases {
            let kind = PatchRotationKind::new(basis, movement).unwrap();
            assert_eq!(PatchRotationMixedSide::for_kind(kind), expected);
        }
    }

    #[test]
    fn spatial_pipe_between_scaled_cubes_matches_cube_height() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(crate::CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("valid scaled cube"),
        );
        graph.add_block(
            Block::new(IVec3::X, BlockKind::Cube(crate::CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("valid scaled cube"),
        );
        let pipe = Pipe::new(IVec3::ZERO, Direction::XPLUS);
        graph.add_pipe(pipe.clone());

        let data = pipe_between_positions_as_gltf_data(IVec3::ZERO, IVec3::X, &pipe, &graph, 1.0);
        let (min, max) = triangle_bounds(&data);

        assert!((min.z + 0.5).abs() < EPS, "unexpected min z: {min:?}");
        assert!((max.z - 2.5).abs() < EPS, "unexpected max z: {max:?}");
    }

    #[test]
    fn standalone_pipe_rendering_handles_distant_coordinate_extremes() {
        let u = Block::new(IVec3::splat(i32::MIN), BlockKind::Port);
        let v = Block::new(IVec3::splat(i32::MAX), BlockKind::Port);
        let pipe = Pipe::new(u.pos(), Direction::XPLUS);
        let graph = BlockGraph::new();

        let positioned = pipe_between_positions_as_gltf_data(u.pos(), v.pos(), &pipe, &graph, 1.0);
        assert!(!positioned.triangles.is_empty());

        let wrapped = pipe_as_gltf_data(&u, &v, &pipe, &graph, 1.0);
        assert!(!wrapped.triangles.is_empty());
    }

    #[test]
    fn stabilizer_visualization_scales_cube_node_surface_height() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(crate::CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("valid scaled cube"),
        );
        let stabilizer = Stabilizer {
            paulis: PauliString::new(0),
            sign: false,
            port_stabilizer: Default::default(),
            interior_nodes: [(IVec3::ZERO, Pauli::Z)].into_iter().collect(),
            interior_edges: Default::default(),
        };
        let generator = StabilizerGenerator::new(stabilizer, StabilizerRowKind::Logical);

        let data = stabilizer_as_gltf_data(&generator, &graph, 1.0)
            .expect("scaled cube stabilizer should render");
        let (min, max) = triangle_bounds(&data);

        assert!((min.z + 0.5).abs() < EPS, "unexpected min z: {min:?}");
        assert!((max.z - 2.5).abs() < EPS, "unexpected max z: {max:?}");
    }

    #[test]
    fn measurement_generator_label_is_derived_at_render_time() {
        let graph = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let stabilizers = graph.stabilizers().expect("T stabilizers");
        let generator = stabilizers
            .iter()
            .find(|generator| generator.is_measurement())
            .expect("T has a measurement generator");

        let data = stabilizer_as_gltf_data(generator, &graph, 2.0)
            .expect("measurement generator should render");

        assert_eq!(data.texts.len(), 1);
        assert_eq!(data.texts[0].text, generator.measurement_name().unwrap());
    }

    #[test]
    fn stabilizer_visualization_scales_spatial_pipe_between_scaled_cubes() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Cube(crate::CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("valid scaled cube"),
        );
        graph.add_block(
            Block::new(IVec3::X, BlockKind::Cube(crate::CubeKind::ZXZ))
                .with_height("2d".parse().expect("valid height"))
                .expect("valid scaled cube"),
        );
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        let stabilizer = Stabilizer {
            paulis: PauliString::new(0),
            sign: false,
            port_stabilizer: Default::default(),
            interior_nodes: Default::default(),
            interior_edges: [((IVec3::ZERO, IVec3::X), Pauli::Z)].into_iter().collect(),
        };
        let generator = StabilizerGenerator::new(stabilizer, StabilizerRowKind::Logical);

        let data = stabilizer_as_gltf_data(&generator, &graph, 1.0)
            .expect("scaled cube pipe stabilizer should render");
        let (min, max) = triangle_bounds(&data);

        assert!((min.z + 0.5).abs() < EPS, "unexpected min z: {min:?}");
        assert!((max.z - 2.5).abs() < EPS, "unexpected max z: {max:?}");
    }

    fn assert_patch_rotation_stabilizer_support(
        basis: Basis,
        movement: IVec2,
        pauli: Pauli,
        expected_shape: Pauli,
    ) {
        let kind = PatchRotationKind::new(basis, movement).unwrap();
        let stride = 3.0;
        let mut data = GltfData::default();
        add_patch_rotation_node_surface(IVec3::ZERO, kind, pauli, stride, &mut data);

        let color = correlation_surface_color(pauli);
        let colors = PatchRotationColors::new(kind);
        let (mixed_x0, mixed_x1) = match colors.mixed_side {
            PatchRotationMixedSide::Left => (-0.5, 0.0),
            PatchRotationMixedSide::Right => (0.0, 0.5),
        };
        let local = |x: f32, y: f32, z: f32| patch_rotation_local_to_global(kind, vec3(x, y, z));

        match expected_shape {
            Pauli::Z => {
                assert_patch_rotation_color_at_named(
                    &data,
                    local(f32::midpoint(mixed_x0, mixed_x1), 1.5, 0.5),
                    color,
                    &format!("{basis:?} {movement:?} {pauli:?} side leg"),
                );
                assert_patch_rotation_color_at_named(
                    &data,
                    local(0.0, 1.0, 0.5),
                    color,
                    &format!("{basis:?} {movement:?} {pauli:?} vertical leg"),
                );
                assert_patch_rotation_color_at_named(
                    &data,
                    local(0.0, 3.0, 2.5),
                    color,
                    &format!("{basis:?} {movement:?} {pauli:?} final line"),
                );
            }
            Pauli::X => {
                assert_patch_rotation_color_at_named(
                    &data,
                    local(0.0, 0.0, 0.0),
                    color,
                    &format!("{basis:?} {movement:?} {pauli:?} initial line"),
                );
                assert_patch_rotation_color_at_named(
                    &data,
                    local(f32::midpoint(mixed_x0, mixed_x1), 1.5, 2.0),
                    color,
                    &format!("{basis:?} {movement:?} {pauli:?} side leg"),
                );
                assert_patch_rotation_color_at_named(
                    &data,
                    local(0.0, 2.0, 2.0),
                    color,
                    &format!("{basis:?} {movement:?} {pauli:?} vertical leg"),
                );
            }
            Pauli::I | Pauli::Y => unreachable!("patch rotation stabilizer shapes are X/Z"),
        }
    }

    #[test]
    fn patch_rotation_stabilizer_visualization_matches_variant_table() {
        for basis in [Basis::X, Basis::Z] {
            // X-axis motion exchanges the construction's two observable shapes;
            // Y-axis motion keeps the basis-labelled shape.
            for (movement, shape) in [
                (ivec2(1, 0), Pauli::Z),
                (ivec2(-1, 0), Pauli::Z),
                (ivec2(0, 1), Pauli::X),
                (ivec2(0, -1), Pauli::X),
            ] {
                assert_patch_rotation_stabilizer_support(
                    basis,
                    movement,
                    Pauli::from(basis),
                    shape,
                );
                assert_patch_rotation_stabilizer_support(
                    basis,
                    movement,
                    Pauli::from(basis.flip()),
                    shape.flip(),
                );
            }
        }
    }

    #[test]
    fn x_plus_y_stabilizer_visualization_matches_observable_evolution() {
        let kind = PatchRotationKind::new(Basis::X, ivec2(0, 1)).unwrap();
        let stride = 3.0;
        let mut z_data = GltfData::default();
        let mut x_data = GltfData::default();
        add_patch_rotation_node_surface(IVec3::ZERO, kind, Pauli::Z, stride, &mut z_data);
        add_patch_rotation_node_surface(IVec3::ZERO, kind, Pauli::X, stride, &mut x_data);

        assert_patch_rotation_color_at_named(
            &z_data,
            vec3(0.0, 1.0, 0.5),
            correlation_surface_color(Pauli::Z),
            "x_plus_y Z input L vertical leg",
        );
        assert_patch_rotation_empty_at_named(
            &z_data,
            vec3(0.0, 2.0, 0.5),
            "x_plus_y Z excluded lower input leg",
        );
        assert_patch_rotation_color_at_named(
            &z_data,
            vec3(0.25, 1.0, 2.5),
            correlation_surface_color(Pauli::Z),
            "x_plus_y Z 3/4 sheet mixed half",
        );
        assert_patch_rotation_empty_at_named(
            &z_data,
            vec3(-0.25, 1.0, 2.5),
            "x_plus_y Z 3/4 sheet excluded half",
        );
        assert_patch_rotation_color_at_named(
            &z_data,
            vec3(0.0, 3.0, 3.0),
            correlation_surface_color(Pauli::Z),
            "x_plus_y Z output midpoint line",
        );
        assert_patch_rotation_empty_at_named(
            &z_data,
            vec3(0.0, 2.75, 3.0),
            "x_plus_y Z excluded output line",
        );

        assert_patch_rotation_color_at_named(
            &x_data,
            vec3(0.0, 0.0, 0.0),
            correlation_surface_color(Pauli::X),
            "x_plus_y X shifted input line",
        );
        assert_patch_rotation_empty_at_named(
            &x_data,
            vec3(0.0, 0.5, 0.0),
            "x_plus_y X excluded input line",
        );
        assert_patch_rotation_color_at_named(
            &x_data,
            vec3(0.0, 0.5, 0.5),
            correlation_surface_color(Pauli::X),
            "x_plus_y X 1/4 sheet input band",
        );
        assert_patch_rotation_empty_at_named(
            &x_data,
            vec3(-0.25, 2.0, 0.5),
            "x_plus_y X 1/4 sheet excluded half",
        );
        assert_patch_rotation_color_at_named(
            &x_data,
            vec3(-0.25, 1.5, 2.0),
            correlation_surface_color(Pauli::X),
            "x_plus_y X final L horizontal leg",
        );
        assert_patch_rotation_color_at_named(
            &x_data,
            vec3(0.0, 2.0, 2.0),
            correlation_surface_color(Pauli::X),
            "x_plus_y X final L vertical leg",
        );
    }

    #[test]
    fn stabilizer_visualization_draws_patch_rotation_node_support() {
        let kind = PatchRotationKind::new(Basis::X, ivec2(1, 0)).unwrap();
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::PatchRotation(kind)));

        let interior_nodes = [(IVec3::ZERO, Pauli::Z)].into_iter().collect();
        let stabilizer = Stabilizer {
            paulis: PauliString::new(0),
            sign: false,
            port_stabilizer: Default::default(),
            interior_nodes,
            interior_edges: Default::default(),
        };
        let generator = StabilizerGenerator::new(stabilizer, StabilizerRowKind::Logical);

        let data = stabilizer_as_gltf_data(&generator, &graph, 3.0)
            .expect("patch rotation stabilizer should render");
        assert_patch_rotation_color_at(&data, vec3(2.0, 0.0, 2.0), RGBA::Z_PURE_BLUE);
    }

    #[test]
    fn stabilizer_visualization_handles_walking_end_pipe() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            IVec3::new(0, 0, 0),
            BlockKind::Walking(
                WalkingKind::new(WalkingBoundaryKind::ZXZ, IVec2::new(1, 1)).unwrap(),
            ),
        ));
        graph.add_block(Block::new(IVec3::new(1, 1, 2), BlockKind::Port));
        graph.add_block(Block::new(IVec3::new(0, 0, -1), BlockKind::Port));
        graph.add_pipe(Pipe::new(IVec3::new(1, 1, 1), Direction::ZPLUS));
        graph.add_pipe(Pipe::new(IVec3::new(0, 0, 0), Direction::ZMINUS));

        let stabilizers = graph.stabilizers().expect("walking graph stabilizers");
        assert!(!stabilizers.is_empty());
        for generator in &stabilizers {
            let mesh = stabilizer_as_gltf_data(generator, &graph, 2.0)
                .expect("walking stabilizer should convert to mesh data");
            assert!(!mesh.triangles.is_empty());
        }
    }

    #[test]
    fn stabilizer_visualization_draws_walking_surface_from_walking_node_support() {
        let graph = parse_blog_to_graph(
            "BLOG 1.0\n\n  0: XZZ [0, 0, 0]\n  1: walk XZZ [0, 0, 1] -> [1, 1, 2]\n  [0, 0, 0] -> +Z\n",
        )
        .expect("walking start-pipe graph should parse");
        let stabilizers = graph.stabilizers().expect("walking graph stabilizers");
        let stride = 3.0;
        let expected_top = IVec3::new(1, 1, 2).as_vec3() * stride + Vec3::Z * 0.5;

        assert!(stabilizers.iter().any(|generator| {
            if !generator
                .stabilizer
                .interior_nodes
                .contains_key(&IVec3::new(0, 0, 1))
            {
                return false;
            }
            let mesh = stabilizer_as_gltf_data(generator, &graph, 2.0)
                .expect("walking stabilizer should convert to mesh data");
            mesh.triangles.values().flatten().flatten().any(|vertex| {
                vertex.abs_diff_eq(expected_top + Vec3::X * 0.5, 1e-5)
                    || vertex.abs_diff_eq(expected_top - Vec3::X * 0.5, 1e-5)
                    || vertex.abs_diff_eq(expected_top + Vec3::Y * 0.5, 1e-5)
                    || vertex.abs_diff_eq(expected_top - Vec3::Y * 0.5, 1e-5)
            })
        }));
    }
}
