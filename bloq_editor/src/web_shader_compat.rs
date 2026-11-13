//! Temporary Bevy 0.20.1 Safari WebGPU compatibility; remove after upstream fixes
//! https://github.com/bevyengine/bevy/issues/26378.
//! Implements the direct texture/sampler arguments from commit
//! bcanfield/bevy@e909449efc07d4d47d8b20e4f3c026439b058d74 without vendoring shaders.
//! Only exact upstream source fingerprints are transformed; no arbitrary WGSL.

#[cfg(target_arch = "wasm32")]
use bevy::{asset::AssetTrackingSystems, prelude::*, shader::Source};

const FIELDS: &[&str] = &[
    "base_color",
    "specular",
    "specular_tint",
    "emissive",
    "metallic_roughness",
    "clearcoat",
    "clearcoat_roughness",
    "specular_transmission",
    "thickness",
    "diffuse_transmission",
    "occlusion",
    "normal_map",
    "clearcoat_normal",
    "anisotropy",
];

fn fingerprint(source: &str) -> u64 {
    source.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

fn statement(lines: &[&str], cursor: &mut usize) -> Option<String> {
    let start = *cursor;
    while !lines.get(*cursor)?.trim_end().ends_with(';') {
        *cursor += 1;
    }
    *cursor += 1;
    Some(lines[start..*cursor].join("\n"))
}

/// Rewrite the known opaque alias stanza and its immediately following sample
/// statements, preserving UV selection and all original material-pass branches.
fn rewrite(source: &str, fields: &[&str], meshlet: bool) -> Option<String> {
    let lines = source.lines().collect::<Vec<_>>();
    let mut cursor = 0;
    let mut output = String::with_capacity(source.len());
    for field in fields {
        let (texture, sampler) = if *field == "clearcoat_normal" {
            ("clearcoat_texture", "clearcoat_sampler")
        } else {
            ("texture", "sampler_")
        };
        let start = (cursor..lines.len().checked_sub(1)?).find(|&index| {
            lines[index].trim() == "@if(BINDLESS)"
                && lines[index + 1]
                    .trim()
                    .starts_with(&format!("let {texture} = bindless_textures_2d["))
        })?;
        output.push_str(&lines[cursor..start].join("\n"));
        output.push('\n');
        let binding = format!("{field}_texture");
        let sampler_binding = format!("{field}_sampler");
        let expected = compact(&format!(
            "@if(BINDLESS) let {texture} = bindless_textures_2d[material_indices[slot].{binding}]; @else let {texture} = pbr_bindings::{binding}; @if(BINDLESS) let {sampler} = bindless_samplers_filtering[material_indices[slot].{sampler_binding}]; @else let {sampler} = pbr_bindings::{sampler_binding};"
        ));
        cursor = start;
        let mut actual = String::new();
        while actual.len() < expected.len() {
            actual.push_str(&compact(lines.get(cursor)?));
            cursor += 1;
        }
        if actual != expected {
            return None;
        }
        if meshlet {
            while lines.get(cursor)?.trim() != "@if(MESHLET_MESH_MATERIAL_PASS)" {
                // Only the known UV coordinate selection may precede sampling.
                if cursor - start > 24 {
                    return None;
                }
                output.push_str(lines[cursor]);
                output.push('\n');
                cursor += 1;
            }
            cursor += 1;
        }
        for gradient in if meshlet {
            &[true, false][..]
        } else {
            &[false][..]
        } {
            let sample = statement(&lines, &mut cursor)?;
            let call = if *gradient {
                "textureSampleGrad("
            } else {
                "textureSampleBias("
            };
            if !sample.contains(call)
                || sample.matches(&format!("{texture},")).count() != 1
                || sample.matches(&format!("{sampler},")).count() != 1
            {
                return None;
            }
            for bindless in [true, false] {
                let condition = match (meshlet, gradient, bindless) {
                    (true, true, true) => "BINDLESS && MESHLET_MESH_MATERIAL_PASS",
                    (true, true, false) => "!BINDLESS && MESHLET_MESH_MATERIAL_PASS",
                    (true, false, true) => "BINDLESS && !MESHLET_MESH_MATERIAL_PASS",
                    (true, false, false) => "!BINDLESS && !MESHLET_MESH_MATERIAL_PASS",
                    (false, _, true) => "BINDLESS",
                    (false, _, false) => "!BINDLESS",
                };
                let (direct_texture, direct_sampler) = if bindless {
                    (
                        format!("bindless_textures_2d[material_indices[slot].{binding}]"),
                        format!(
                            "bindless_samplers_filtering[material_indices[slot].{sampler_binding}]"
                        ),
                    )
                } else {
                    (
                        format!("pbr_bindings::{binding}"),
                        format!("pbr_bindings::{sampler_binding}"),
                    )
                };
                let direct = sample
                    .replace(&format!("{texture},"), &format!("{direct_texture},"))
                    .replace(&format!("{sampler},"), &format!("{direct_sampler},"));
                output.push_str(&format!("@if({condition})\n{direct}\n"));
            }
            if *gradient {
                if !matches!(
                    lines.get(cursor)?.trim(),
                    "@else" | "@if(!MESHLET_MESH_MATERIAL_PASS)"
                ) {
                    return None;
                }
                cursor += 1;
            }
        }
    }
    output.push_str(&lines[cursor..].join("\n"));
    Some(output)
}

fn replacement(path: &str, source: &str) -> Option<String> {
    let (hash, fields, meshlet) = match path {
        "embedded://bevy_pbr/render/pbr_fragment.wesl" => (0xa72dbf3cd15588d6, FIELDS, true),
        "embedded://bevy_pbr/render/pbr_prepass.wesl" => {
            (0x07c22c70e879b290, &["normal_map"][..], true)
        }
        "embedded://bevy_pbr/render/pbr_prepass_functions.wesl" => {
            (0xe77aec3d0759e1fd, &["base_color"][..], false)
        }
        _ => return None,
    };
    (fingerprint(source) == hash)
        .then(|| rewrite(source, fields, meshlet))
        .flatten()
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn install(app: &mut App) {
    // Loaded shaders enter Assets in PreUpdate. Patch before PostUpdate emits
    // asset events, so the render world never extracts the invalid first version.
    app.add_systems(
        PreUpdate,
        patch_loaded_shaders
            .after(AssetTrackingSystems)
            .run_if(resource_changed::<Assets<Shader>>),
    );
}

#[cfg(target_arch = "wasm32")]
fn patch_loaded_shaders(mut shaders: ResMut<Assets<Shader>>) {
    let changes = shaders
        .iter()
        .filter_map(|(id, shader)| {
            let Source::Wesl(source) = &shader.source else {
                return None;
            };
            replacement(&shader.path, source).map(|source| (id, source))
        })
        .collect::<Vec<_>>();
    for (id, source) in changes {
        // Imports, shader definitions, handles and validation settings are unchanged.
        shaders
            .get_mut(id)
            .expect("shader was just inspected")
            .source = Source::Wesl(source.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_sources_and_paths_are_never_rewritten() {
        assert!(replacement("user.wesl", "let texture = something;").is_none());
        assert!(
            replacement(
                "embedded://bevy_pbr/render/pbr_fragment.wesl",
                "changed upstream shader"
            )
            .is_none()
        );
        assert!(
            rewrite(
                "@if(BINDLESS)\nlet texture = bindless_textures_2d[wrong];",
                &["base_color"],
                true
            )
            .is_none()
        );
    }

    fn aliases() -> &'static str {
        "@if(BINDLESS)\nlet texture = bindless_textures_2d[material_indices[slot].base_color_texture];\n@else\nlet texture = pbr_bindings::base_color_texture;\n@if(BINDLESS)\nlet sampler_ = bindless_samplers_filtering[material_indices[slot].base_color_sampler];\n@else\nlet sampler_ = pbr_bindings::base_color_sampler;\n"
    }

    #[test]
    fn material_sampling_preserves_uv_and_meshlet_branches() {
        let source = format!(
            "{}@if(UV_B)\nlet coords = uv_b;\n@else\nlet coords = uv;\n@if(MESHLET_MESH_MATERIAL_PASS)\ncolor *= textureSampleGrad(texture, sampler_, coords, dx, dy);\n@else\ncolor *= textureSampleBias(texture, sampler_, coords, bias);",
            aliases()
        );
        let fixed = rewrite(&source, &["base_color"], true).unwrap();
        assert!(fixed.contains("@if(UV_B)\nlet coords = uv_b;\n@else\nlet coords = uv;"));
        assert_eq!(fixed.matches("textureSampleGrad(").count(), 2);
        assert_eq!(fixed.matches("textureSampleBias(").count(), 2);
        assert!(fixed.contains("@if(!BINDLESS && !MESHLET_MESH_MATERIAL_PASS)"));
        assert!(fixed.contains("textureSampleGrad(pbr_bindings::base_color_texture, pbr_bindings::base_color_sampler, coords, dx, dy)"));
        assert!(!fixed.contains("let texture"));
    }

    #[test]
    fn prepass_multiline_sampling_retains_both_binding_modes() {
        let source = format!(
            "{}output_color *= textureSampleBias(\n texture,\n sampler_,\n uv,\n bias\n);",
            aliases()
        );
        let fixed = rewrite(&source, &["base_color"], false).unwrap();
        assert_eq!(fixed.matches("textureSampleBias(").count(), 2);
        assert!(fixed.contains("pbr_bindings::base_color_sampler,"));
        assert!(
            fixed.contains(
                "bindless_samplers_filtering[material_indices[slot].base_color_sampler],"
            )
        );
        assert!(!fixed.contains("let sampler_"));
    }

    #[test]
    #[ignore = "manual check against installed Bevy source; set BLOQ_BEVY_PBR_SOURCE"]
    fn installed_upstream_sources_match_all_guarded_rewrites() {
        let root = std::env::var("BLOQ_BEVY_PBR_SOURCE").unwrap();
        for name in ["pbr_fragment", "pbr_prepass", "pbr_prepass_functions"] {
            let source = std::fs::read_to_string(format!("{root}/{name}.wesl")).unwrap();
            let path = format!("embedded://bevy_pbr/render/{name}.wesl");
            let fixed = replacement(&path, &source).expect("pinned Bevy shader must match");
            assert!(!fixed.contains("let texture ="));
            assert!(!fixed.contains("let sampler_ ="));
            assert!(!fixed.contains("let clearcoat_texture ="));
            assert!(replacement(&path, &fixed).is_none());
        }
    }
}
