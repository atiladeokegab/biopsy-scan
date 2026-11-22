// Preflight workflow palette, by tma.json type_names index and type_groups: tumour in coral tints (#FF7A59), immune in
// amber/gold (#F5C451), stroma and other in slate (#56687A); emerald (--accent) is kept for highlights. Lightness varies
// within each family so neighbouring types stay apart. Marker channels keep the pseudo-fluorescence colours.
export const TYPE_COLORS = [
  "#F5C451", "#9AAFC2", "#FFE08A", "#D99A1E", "#FFB347", "#A87A1F", "#56687A", "#6F8599", // B, vessel, CD4, CD8, gran, mac, other, stroma
  "#FF7A59", "#FFB39E", "#D94B33", "#FF5E6C", "#F28C7E", "#B83A2A", "#FFD0C2"];          // tumour 1-7
export const MARKER_COLORS = ["#3d7bff", "#2bff88", "#ff3df0", "#ffd23d", "#ff5a3d", "#3dfff2"];
export const DAB = "#8b5a2b";          // simulated brown DAB
export const HAEMATOXYLIN = "#5a6fb0"; // simulated blue counterstain
