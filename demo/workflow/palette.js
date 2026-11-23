// Biopsy Scan workflow palette, in the design system's families (tokens.css): tumour types are coral tints of --tumour
// #FF7A59, immune types amber/gold tints of --immune #F5C451, stroma/vessel/other slate tints of --stroma #56687A,
// told apart within a family by lightness and saturation. Emerald --accent #2BD4A4 is for highlights only: never a
// cell type. Index = tma.json type_names index.
export const TYPE_COLORS = [
  "#FFE3A1", // B cell            immune: pale gold
  "#C3D3E4", // Blood vessel      stroma: lightest slate
  "#F5C451", // CD4 T cell        immune: --immune
  "#E39B1F", // CD8 T cell        immune: deep amber
  "#FFD27A", // Granulocyte       immune: soft gold
  "#C98A2E", // Macrophage        immune: dark ochre
  "#7D93AB", // Other             stroma: bright slate (reads on the #05070B ground)
  "#9DB2C8", // Stroma            stroma: pale slate
  "#FF7A59", // Tumor 1           tumour: --tumour
  "#FF4F2B", // Tumor 2 (Ki67)    tumour: saturated coral (proliferating)
  "#FFA58E", // Tumor 3           tumour: light coral
  "#E2603F", // Tumor 4           tumour: deep coral
  "#FFC6B6", // Tumor 5           tumour: pale coral
  "#FF8F6B", // Tumor 6 / DC      tumour: warm coral
  "#C9553A", // Tumor 7           tumour: brick coral
];
// Channels, tma.json markers order: DAPI, PanCK, CD8, CD68, aSMA, CD20. DAPI is pale ink #F3EFE6 at ~35% alpha,
// pre-multiplied on the black ground (the views add channels with "lighter", so this equals the alpha) and kept as
// 6-digit hex because the views parse and suffix these strings.
export const DAPI_INK = "#F3EFE6", DAPI_ALPHA = 0.35;
export const MARKER_COLORS = ["#555450", "#FF7A59", "#F5C451", "#E39B1F", "#8FA3B8", "#FFE3A1"];
export const DAB = "#8b5a2b";          // simulated brown DAB
export const HAEMATOXYLIN = "#5a6fb0"; // simulated blue counterstain
