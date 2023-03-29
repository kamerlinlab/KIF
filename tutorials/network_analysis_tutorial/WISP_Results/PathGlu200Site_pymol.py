# To run this script you will need to get a copy of 'draw_links.py' 
# You can find it freely available here: 
# http://pldserver1.biochem.queensu.ca/~rlc/work/pymol/draw_links.py 
# Place the 'draw_links.py' file in your working directory. 
run draw_links.py
sele path_residues, resi 199+287+196+192+151+176+178+180+291+294+188+190+220+296+150+152+182+195+189+265+290+179+181+183+262+ 
show sticks, path_residues 
show spheres, resi 262 and name CA
set sphere_scale, 0.0033, resi 262 and name CA
show spheres, resi 265 and name CA
set sphere_scale, 0.3833, resi 265 and name CA
show spheres, resi 150 and name CA
set sphere_scale, 0.0900, resi 150 and name CA
show spheres, resi 151 and name CA
set sphere_scale, 0.4000, resi 151 and name CA
show spheres, resi 152 and name CA
set sphere_scale, 0.0933, resi 152 and name CA
show spheres, resi 287 and name CA
set sphere_scale, 0.7900, resi 287 and name CA
show spheres, resi 290 and name CA
set sphere_scale, 0.0467, resi 290 and name CA
show spheres, resi 291 and name CA
set sphere_scale, 0.1433, resi 291 and name CA
show spheres, resi 294 and name CA
set sphere_scale, 0.5100, resi 294 and name CA
show spheres, resi 296 and name CA
set sphere_scale, 0.3400, resi 296 and name CA
show spheres, resi 176 and name CA
set sphere_scale, 0.3133, resi 176 and name CA
show spheres, resi 178 and name CA
set sphere_scale, 0.5667, resi 178 and name CA
show spheres, resi 179 and name CA
set sphere_scale, 0.2767, resi 179 and name CA
show spheres, resi 180 and name CA
set sphere_scale, 1.0000, resi 180 and name CA
show spheres, resi 181 and name CA
set sphere_scale, 0.0167, resi 181 and name CA
show spheres, resi 182 and name CA
set sphere_scale, 0.4367, resi 182 and name CA
show spheres, resi 183 and name CA
set sphere_scale, 0.0067, resi 183 and name CA
show spheres, resi 188 and name CA
set sphere_scale, 0.6867, resi 188 and name CA
show spheres, resi 189 and name CA
set sphere_scale, 0.2433, resi 189 and name CA
show spheres, resi 190 and name CA
set sphere_scale, 0.6867, resi 190 and name CA
show spheres, resi 192 and name CA
set sphere_scale, 0.5633, resi 192 and name CA
show spheres, resi 195 and name CA
set sphere_scale, 0.2100, resi 195 and name CA
show spheres, resi 196 and name CA
set sphere_scale, 0.8100, resi 196 and name CA
show spheres, resi 199 and name CA
set sphere_scale, 1.0000, resi 199 and name CA
show spheres, resi 220 and name CA
set sphere_scale, 0.6033, resi 220 and name CA
draw_links selection1=resi 199, selection2=resi 287, color=grey, radius=0.5 
draw_links selection1=resi 287, selection2=resi 196, color=grey, radius=0.4093 
draw_links selection1=resi 196, selection2=resi 192, color=grey, radius=0.3122 
draw_links selection1=resi 192, selection2=resi 151, color=grey, radius=0.2131 
draw_links selection1=resi 151, selection2=resi 176, color=grey, radius=0.1392 
draw_links selection1=resi 176, selection2=resi 178, color=grey, radius=0.1983 
draw_links selection1=resi 178, selection2=resi 180, color=grey, radius=0.0928 
draw_links selection1=resi 287, selection2=resi 291, color=grey, radius=0.0907 
draw_links selection1=resi 291, selection2=resi 294, color=grey, radius=0.0907 
draw_links selection1=resi 294, selection2=resi 188, color=grey, radius=0.3228 
draw_links selection1=resi 188, selection2=resi 190, color=grey, radius=0.2806 
draw_links selection1=resi 190, selection2=resi 220, color=grey, radius=0.2954 
draw_links selection1=resi 220, selection2=resi 180, color=grey, radius=0.1118 
draw_links selection1=resi 196, selection2=resi 296, color=grey, radius=0.2004 
draw_links selection1=resi 296, selection2=resi 150, color=grey, radius=0.057 
draw_links selection1=resi 150, selection2=resi 152, color=grey, radius=0.057 
draw_links selection1=resi 152, selection2=resi 176, color=grey, radius=0.0591 
draw_links selection1=resi 192, selection2=resi 294, color=grey, radius=0.1287 
draw_links selection1=resi 151, selection2=resi 188, color=grey, radius=0.1118 
draw_links selection1=resi 178, selection2=resi 182, color=grey, radius=0.1097 
draw_links selection1=resi 182, selection2=resi 180, color=grey, radius=0.1456 
draw_links selection1=resi 199, selection2=resi 195, color=grey, radius=0.1329 
draw_links selection1=resi 195, selection2=resi 196, color=grey, radius=0.1034 
draw_links selection1=resi 296, selection2=resi 294, color=grey, radius=0.0738 
draw_links selection1=resi 190, selection2=resi 178, color=grey, radius=0.1392 
draw_links selection1=resi 188, selection2=resi 189, color=grey, radius=0.154 
draw_links selection1=resi 189, selection2=resi 190, color=grey, radius=0.154 
draw_links selection1=resi 178, selection2=resi 265, color=grey, radius=0.0633 
draw_links selection1=resi 265, selection2=resi 180, color=grey, radius=0.1561 
draw_links selection1=resi 296, selection2=resi 151, color=grey, radius=0.0401 
draw_links selection1=resi 195, selection2=resi 290, color=grey, radius=0.0295 
draw_links selection1=resi 290, selection2=resi 294, color=grey, radius=0.0295 
draw_links selection1=resi 296, selection2=resi 192, color=grey, radius=0.0443 
draw_links selection1=resi 220, selection2=resi 265, color=grey, radius=0.0865 
draw_links selection1=resi 220, selection2=resi 182, color=grey, radius=0.0907 
draw_links selection1=resi 178, selection2=resi 220, color=grey, radius=0.0422 
draw_links selection1=resi 178, selection2=resi 179, color=grey, radius=0.0464 
draw_links selection1=resi 179, selection2=resi 180, color=grey, radius=0.1181 
draw_links selection1=resi 182, selection2=resi 265, color=grey, radius=0.0633 
draw_links selection1=resi 220, selection2=resi 179, color=grey, radius=0.0738 
draw_links selection1=resi 182, selection2=resi 220, color=grey, radius=0.0211 
draw_links selection1=resi 265, selection2=resi 182, color=grey, radius=0.0506 
draw_links selection1=resi 182, selection2=resi 179, color=grey, radius=0.0338 
draw_links selection1=resi 192, selection2=resi 296, color=grey, radius=0.0148 
draw_links selection1=resi 265, selection2=resi 220, color=grey, radius=0.0127 
draw_links selection1=resi 179, selection2=resi 220, color=grey, radius=0.0105 
draw_links selection1=resi 179, selection2=resi 182, color=grey, radius=0.0232 
draw_links selection1=resi 151, selection2=resi 152, color=grey, radius=0.0021 
draw_links selection1=resi 265, selection2=resi 179, color=grey, radius=0.0211 
draw_links selection1=resi 179, selection2=resi 265, color=grey, radius=0.0232 
draw_links selection1=resi 220, selection2=resi 178, color=grey, radius=0.0148 
draw_links selection1=resi 182, selection2=resi 181, color=grey, radius=0.0042 
draw_links selection1=resi 181, selection2=resi 180, color=grey, radius=0.0084 
draw_links selection1=resi 182, selection2=resi 178, color=grey, radius=0.0063 
draw_links selection1=resi 178, selection2=resi 183, color=grey, radius=0.0042 
draw_links selection1=resi 183, selection2=resi 182, color=grey, radius=0.0021 
draw_links selection1=resi 182, selection2=resi 262, color=grey, radius=0.0021 
draw_links selection1=resi 262, selection2=resi 265, color=grey, radius=0.0021 
draw_links selection1=resi 181, selection2=resi 265, color=grey, radius=0.0021 
draw_links selection1=resi 183, selection2=resi 265, color=grey, radius=0.0021 
draw_links selection1=resi 220, selection2=resi 181, color=grey, radius=0.0042 
draw_links selection1=resi 265, selection2=resi 181, color=grey, radius=0.0021 
group Paths, link*