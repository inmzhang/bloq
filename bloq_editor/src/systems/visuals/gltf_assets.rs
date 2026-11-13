//! glTF geometry parsing, spawning of mesh/line/text child entities, and
//! pick-proxy mesh generation.

use super::*;

pub(super) fn update_cached_mesh_part_materials(
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
    render_cache: &mut RenderAssetCache,
    data: &GltfData,
    mesh_parts: &RenderedMeshParts,
    alpha_multiplier: f32,
    face_material_override: Option<Handle<StandardMaterial>>,
    display_material_override: Option<Handle<StandardMaterial>>,
    triangle_style: MaterialStyle,
) {
    for (entity, (rgba, _)) in mesh_parts.triangles.iter().zip(data.triangles.iter()) {
        let mut color = rgba.to_f32_array();
        color[3] *= alpha_multiplier;
        let material = get_or_insert_material(render_cache, materials, color, triangle_style);
        let resting_material = face_material_override
            .clone()
            .unwrap_or_else(|| material.clone());
        let display_material = display_material_override
            .clone()
            .unwrap_or_else(|| resting_material.clone());
        commands.entity(*entity).insert((
            MeshMaterial3d(display_material),
            OriginalMaterial(resting_material),
        ));
    }

    let line_style = MaterialStyle {
        unlit: true,
        double_sided: false,
        cull_none: false,
    };
    for (entity, (rgba, _)) in mesh_parts.lines.iter().zip(data.lines.iter()) {
        let mut color = rgba.to_f32_array();
        color[3] *= alpha_multiplier;
        let material = get_or_insert_material(render_cache, materials, color, line_style);
        // Module ownership tints faces only; keeping the authored outline
        // preserves each block's silhouette and Hadamard seam marks.
        let resting_material = material.clone();
        let display_material = display_material_override
            .clone()
            .unwrap_or_else(|| resting_material.clone());
        commands.entity(*entity).insert((
            MeshMaterial3d(display_material),
            OriginalMaterial(resting_material),
        ));
    }
}

pub(super) struct GltfGeometryAnalysis {
    pub(super) pick_proxy: PickProxyBounds,
    pub(super) mesh_keys: CachedMeshKeys,
}

#[derive(Clone, Copy)]
pub(super) struct PickProxyBounds {
    pub(super) min: Vec3,
    pub(super) max: Vec3,
}

pub(super) fn analyze_gltf_geometry(data: &GltfData) -> GltfGeometryAnalysis {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut triangle_mesh_keys = Vec::with_capacity(data.triangles.len());
    let mut line_mesh_keys = Vec::with_capacity(data.lines.len());
    for triangles in data.triangles.values() {
        let mut mesh_hasher = DefaultHasher::new();
        hash_topology(PrimitiveTopology::TriangleList, &mut mesh_hasher);
        for triangle in triangles {
            for point in triangle {
                hash_vec3_position(*point, &mut mesh_hasher);
                min = min.min(*point);
                max = max.max(*point);
            }
        }
        triangle_mesh_keys.push(mesh_hasher.finish());
    }
    for lines in data.lines.values() {
        let mut mesh_hasher = DefaultHasher::new();
        hash_topology(PrimitiveTopology::LineList, &mut mesh_hasher);
        for line in lines {
            for point in line {
                hash_vec3_position(*point, &mut mesh_hasher);
                min = min.min(*point);
                max = max.max(*point);
            }
        }
        line_mesh_keys.push(mesh_hasher.finish());
    }
    for label in &data.texts {
        min = min.min(label.position - Vec3::splat(0.25));
        max = max.max(label.position + Vec3::splat(0.25));
    }

    let pick_proxy = if min.is_finite() && max.is_finite() {
        let padding = Vec3::splat(0.08);
        PickProxyBounds {
            min: min - padding,
            max: max + padding,
        }
    } else {
        PickProxyBounds {
            min: Vec3::splat(-0.5),
            max: Vec3::splat(0.5),
        }
    };
    GltfGeometryAnalysis {
        pick_proxy,
        mesh_keys: CachedMeshKeys {
            triangles: triangle_mesh_keys,
            lines: line_mesh_keys,
        },
    }
}

pub(super) fn pick_box_positions(min: Vec3, max: Vec3) -> Vec<[f32; 3]> {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    .into_iter()
    .flat_map(|(a, b, c)| {
        [
            corners[a].to_array(),
            corners[b].to_array(),
            corners[c].to_array(),
        ]
    })
    .collect()
}

pub(super) fn get_or_insert_pick_proxy_mesh(
    render_cache: &mut RenderAssetCache,
    meshes: &mut Assets<Mesh>,
    min: Vec3,
    max: Vec3,
) -> Handle<Mesh> {
    let key = PickProxyCacheKey {
        min_bits: [min.x.to_bits(), min.y.to_bits(), min.z.to_bits()],
        max_bits: [max.x.to_bits(), max.y.to_bits(), max.z.to_bits()],
    };
    if let Some(existing) = render_cache.pick_proxy_meshes.get(&key) {
        return existing.clone();
    }

    let positions = pick_box_positions(min, max);
    let handle = get_or_insert_triangle_mesh(render_cache, meshes, &positions);
    render_cache.pick_proxy_meshes.insert(key, handle.clone());
    handle
}

/// Spawns the triangle-mesh, line-mesh, and text-label child entities for a
/// piece of glTF geometry under `parent`, reusing cached meshes and materials.
/// Used by both the live scene and the thumbnail previews.
pub(crate) fn spawn_gltf_data_parts(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    render_cache: &mut RenderAssetCache,
    parent: Entity,
    data: GltfData,
    mesh_keys: Option<&CachedMeshKeys>,
    alpha_multiplier: f32,
    face_material_override: Option<Handle<StandardMaterial>>,
    display_material_override: Option<Handle<StandardMaterial>>,
    render_layers: Option<RenderLayers>,
    pickable: Pickable,
) {
    let _ = spawn_gltf_data_parts_ref(
        commands,
        meshes,
        materials,
        render_cache,
        parent,
        &data,
        mesh_keys,
        alpha_multiplier,
        face_material_override,
        display_material_override,
        render_layers,
        pickable,
    );
}

pub(super) fn spawn_gltf_data_parts_ref(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    render_cache: &mut RenderAssetCache,
    parent: Entity,
    data: &GltfData,
    mesh_keys: Option<&CachedMeshKeys>,
    alpha_multiplier: f32,
    face_material_override: Option<Handle<StandardMaterial>>,
    display_material_override: Option<Handle<StandardMaterial>>,
    render_layers: Option<RenderLayers>,
    pickable: Pickable,
) -> RenderedMeshParts {
    let mut parts = RenderedMeshParts::default();
    for (index, (rgba, triangles)) in data.triangles.iter().enumerate() {
        let mesh_key = mesh_keys
            .and_then(|keys| keys.triangles.get(index))
            .copied();
        let mesh =
            get_or_insert_triangle_mesh_from_triangles(render_cache, meshes, triangles, mesh_key);

        let mut color = rgba.to_f32_array();
        color[3] *= alpha_multiplier;
        let material = get_or_insert_material(
            render_cache,
            materials,
            color,
            editor_triangle_material_style(),
        );

        let resting_material = face_material_override
            .clone()
            .unwrap_or_else(|| material.clone());
        let display_material = display_material_override
            .clone()
            .unwrap_or_else(|| resting_material.clone());
        let child = spawn_mesh_part(
            commands,
            parent,
            mesh,
            display_material,
            resting_material,
            render_layers.as_ref(),
            &pickable,
        );
        parts.triangles.push(child);
    }

    for (index, (rgba, lines)) in data.lines.iter().enumerate() {
        let mesh_key = mesh_keys.and_then(|keys| keys.lines.get(index)).copied();
        let mesh = get_or_insert_line_mesh_from_lines(render_cache, meshes, lines, mesh_key);

        let mut color = rgba.to_f32_array();
        color[3] *= alpha_multiplier;
        let material = get_or_insert_material(
            render_cache,
            materials,
            color,
            MaterialStyle {
                unlit: true,
                double_sided: false,
                cull_none: false,
            },
        );

        let resting_material = material.clone();
        let display_material = display_material_override
            .clone()
            .unwrap_or_else(|| resting_material.clone());
        let child = spawn_mesh_part(
            commands,
            parent,
            mesh,
            display_material,
            resting_material,
            render_layers.as_ref(),
            &pickable,
        );
        parts.lines.push(child);
    }

    for label in &data.texts {
        let (pos, text, rgba) = (&label.position, &label.text, &label.color);
        let text_material = get_or_insert_text_material(render_cache, materials);
        commands.entity(parent).with_children(|parent| {
            let mut child = parent.spawn((
                Text3d::new(text.clone()),
                text_label_style(Srgba::from_f32_array(rgba.to_f32_array()), 0.25),
                Transform::from_translation(*pos),
                Mesh3d::default(),
                MeshMaterial3d(text_material),
                pickable,
            ));
            if let Some(layers) = render_layers.as_ref() {
                child.insert(layers.clone());
            }
        });
    }
    parts
}

pub(super) fn text_label_style(color: Srgba, world_scale: f32) -> Text3dStyling {
    Text3dStyling {
        size: 10.,
        font: "Zed Mono".into(),
        stroke: NonZero::new(10),
        color,
        stroke_color: Srgba::BLACK,
        layer_offset: 0.001,
        world_scale: Some(Vec2::splat(world_scale)),
        ..Default::default()
    }
}

fn spawn_mesh_part(
    commands: &mut Commands,
    parent: Entity,
    mesh: Handle<Mesh>,
    display_material: Handle<StandardMaterial>,
    original_material: Handle<StandardMaterial>,
    render_layers: Option<&RenderLayers>,
    pickable: &Pickable,
) -> Entity {
    let mut child = commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(display_material),
        OriginalMaterial(original_material),
        NotShadowCaster,
        *pickable,
        ChildOf(parent),
    ));
    if let Some(layers) = render_layers {
        child.insert(layers.clone());
    }
    child.id()
}

#[derive(Clone, Copy)]
pub(super) struct MaterialStyle {
    pub(super) unlit: bool,
    pub(super) double_sided: bool,
    pub(super) cull_none: bool,
}

pub(super) const fn editor_triangle_material_style() -> MaterialStyle {
    MaterialStyle {
        unlit: false,
        double_sided: true,
        cull_none: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_part_bundle_preserves_materials_picking_and_required_components() {
        let mut world = World::new();
        // VisibilityPlugin registers this requirement in the running editor.
        world.register_required_components::<Mesh3d, Visibility>();
        let mut materials = Assets::<StandardMaterial>::default();
        let display = materials.add(StandardMaterial::default());
        let original = materials.add(StandardMaterial::default());
        let parent = world.spawn_empty().id();
        for layers in [None, Some(RenderLayers::layer(3))] {
            let entity = spawn_mesh_part(
                &mut world.commands(),
                parent,
                Handle::default(),
                display.clone(),
                original.clone(),
                layers.as_ref(),
                &Pickable::IGNORE,
            );
            world.flush();
            let part = world.entity(entity);
            assert_eq!(
                part.get::<MeshMaterial3d<StandardMaterial>>().unwrap().0,
                display
            );
            assert_eq!(part.get::<OriginalMaterial>().unwrap().0, original);
            assert!(part.contains::<NotShadowCaster>());
            assert!(part.contains::<Transform>());
            assert!(part.contains::<Visibility>());
            let pickable = part.get::<Pickable>().unwrap();
            assert!(!pickable.is_hoverable && !pickable.should_block_lower);
            assert_eq!(part.get::<RenderLayers>(), layers.as_ref());
            assert_eq!(part.get::<ChildOf>().unwrap().parent(), parent);
            assert!(
                world
                    .entity(parent)
                    .get::<Children>()
                    .unwrap()
                    .contains(&entity)
            );
        }
        let children = world.entity(parent).get::<Children>().unwrap().to_vec();
        world.despawn(parent);
        assert!(
            children
                .iter()
                .all(|child| world.get_entity(*child).is_err())
        );
    }
}
