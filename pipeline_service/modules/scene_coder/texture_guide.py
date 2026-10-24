"""[pgh16 R78] Our coder system-prompt section: generated surface maps (DataTexture) with the exact three.js names.

Why: UtevoLux writes texture code often; with pgh7's prompt (no texture section) it invents constant names that do not
exist (RepeatWrap, LinearMipmapipmap, UnsignedByteTypeFormat, LinearSpace, ...), the validator rejects the module
(UNKNOWN_THREE_API) and the draft is lost. This section states the real names once.

Used only by the prompt variants "ours" / "oursneg" of coder_prompt.py (env PGH_CODER_PROMPT; the default is "es").
The text is original (written for pgh16 from the three.js r183/r185 sources and our own js_checker); every THREE.<name>
in it is checked against the three.js build exports and js_checker/validator/threeAllowlist.js (pgh16/tests).
"neg" adds a list of non-existent names that UtevoLux produced on the box (utevo_diag/utevo_diag2 checker logs) - a
test variant: naming wrong tokens may prime them.
"""
from __future__ import annotations

_TEXTURE_GUIDE_BODY = """\
Generated surface maps (DataTexture) - exact API for this three.js build:

Decide first. Most objects need NO texture: flat paint, plastic, glass and
plain metal are best as a material color plus roughness/metalness. A map is
worth its cost only for a fine pattern that repeats across a large surface and
would otherwise need hundreds of meshes: wood grain, woven fabric, knit,
brick or tile with grout lines, stripes, checks, carbon weave, stitched
leather, speckled stone, rust or wear blotches. Big distinct features (panels,
buttons, labels, painted motifs) stay geometry or separate materials, as the
handbooks above say. Get the shape right first; a texture never fixes a wrong
silhouette. Use 1 to 3 textures per object at most.

The only texture source here is `new THREE.DataTexture(...)` filled from a
typed array inside `generate`. There is no canvas and no image file:
CanvasTexture, Texture, TextureLoader, any *Loader, document and
OffscreenCanvas are all rejected by the validator.

Recipe (deterministic, no Math.random):

  function makeMap(THREE, size, shade) {
    const px = new Uint8Array(size * size * 4);      // 4 bytes per texel: R,G,B,A
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const c = shade(x / size, y / size);       // returns [r, g, b], 0..255
        const o = (y * size + x) * 4;
        px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
      }
    }
    const tex = new THREE.DataTexture(px, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.generateMipmaps = true;
    tex.needsUpdate = true;
    return tex;
  }
  const grainTex = makeMap(THREE, 128, (u, v) => { /* pattern */ return [150, 105, 60]; });
  grainTex.colorSpace = THREE.SRGBColorSpace;      // a color map
  grainTex.repeat.set(2, 1);
  const woodMat = new THREE.MeshStandardMaterial({ map: grainTex, color: 0xffffff, roughness: 0.7, metalness: 0.0 });

Exact names (copy them letter for letter; anything else fails the module):
- wrapping (wrapS / wrapT): THREE.RepeatWrapping, THREE.ClampToEdgeWrapping,
  THREE.MirroredRepeatWrapping
- magFilter: THREE.LinearFilter or THREE.NearestFilter
- minFilter: THREE.LinearFilter, THREE.NearestFilter,
  THREE.LinearMipmapLinearFilter, THREE.LinearMipmapNearestFilter,
  THREE.NearestMipmapLinearFilter, THREE.NearestMipmapNearestFilter
- format: THREE.RGBAFormat (with a Uint8Array of width*height*4 bytes).
  type: THREE.UnsignedByteType. A type is not a format: never combine them
  into one name.
- colorSpace (a texture property, set by assignment): THREE.SRGBColorSpace,
  THREE.LinearSRGBColorSpace, THREE.NoColorSpace
@@NEG@@
Rules that silently break a map (the validator does not catch them):
- `tex.needsUpdate = true` after the data is filled; without it the map is
  never uploaded and the surface renders black.
- DataTexture starts with NearestFilter for both filters and
  generateMipmaps = false. A Mipmap minFilter needs `generateMipmaps = true`;
  otherwise keep minFilter = THREE.LinearFilter. Use power-of-two sizes
  (64, 128, 256).
- The array must hold exactly width * height * 4 bytes for RGBAFormat.
  The first row of the array is the bottom of the image (v = 0; flipY is off
  for data textures).
- colorSpace: THREE.SRGBColorSpace for `map` and `emissiveMap` (colors);
  leave THREE.NoColorSpace (the default) on data maps: roughnessMap,
  metalnessMap, aoMap, bumpMap, alphaMap. Give each role its own texture.
- A map multiplies the material value: `map` x `color` (use color 0xffffff to
  show the map as painted), roughnessMap reads the G channel x `roughness`,
  metalnessMap reads the B channel x `metalness`, aoMap reads the R channel.
- `repeat` and `offset` live on the texture, so every material sharing it
  tiles the same way; `repeat` only tiles with RepeatWrapping on both axes.
- Built-in geometries carry UVs. A custom BufferGeometry needs its own
  `uv` attribute (`geom.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))`),
  or the whole surface samples a single texel.
- Prefer bumpMap (grayscale height) for small relief such as grain, weave or
  stitching; normalMap data is easy to get wrong.

Budget: keep the DataTexture data of the whole module within 1 MB (the spec
limit above): 128x128 RGBA = 64 KB, 256x256 = 256 KB, 512x512 = 1 MB. The
validator rejects a module over 4 MiB (TEXTURE_DATA_EXCEEDED); each texture
object is counted, so share one texture between meshes instead of cloning it
(a clone is counted again). Fill pixels with loops, never with literal
arrays (50 KB literal budget), and keep the per-texel math light (5 s limit).
"""

TEXTURE_GUIDE_NEG_NAMES = """\
These do NOT exist and are rejected: RepeatWrap, RepeatWrapper, Repeat,
LinearMipmap, LinearMipmapipmap, MipmapNearestFilter, AnisotropicFilter, RGBA,
RGBA8, RGBA8Format, UnsignedByteTypeFormat, UNSIGNED_BYTE, LinearSpace,
LinearColorSpace, LinearSpaceColorSpace, ColorSpace, NoneColorSpace,
sRGBEncoding, LinearEncoding, and the old `texture.encoding` property.
"""

TEXTURE_GUIDE = _TEXTURE_GUIDE_BODY.replace("@@NEG@@", "")


def texture_guide_text(neg: bool = False) -> str:
    """The section text (with the non-existent-name list when neg)."""
    return _TEXTURE_GUIDE_BODY.replace("@@NEG@@", TEXTURE_GUIDE_NEG_NAMES if neg else "")
