


top="/mnt/var1/1-2024-2025/2-横向-ALP-李迪-何晓-刘迎亚-翟纪航/3-newrun/15E-MC/3_md/200ps-nvt-0.5fs/model.prmtop"
traj="/mnt/var1/1-2024-2025/2-横向-ALP-李迪-何晓-刘迎亚-翟纪航/3-newrun/15E-MC/3_md/200ps-nvt-0.5fs/md-b10w-50ps.ncdf"

#Iw_dipole_derivative.v7.py -top ${top} -ncdf ${traj} \
#						   -dt 0.0005 -SES "0:80000:1" -Oe -0.84 -He 0.42 -frac 0.1 \
#						   -outf qv_autocoor_40ps.csv -seles "(around 4 (resid 1-898) ) and type OW" \
#						   -hann False

Iw_dipole_derivative.v8.py -top ${top} -ncdf ${traj} \
						   -dt 0.0005 -SES "0:80000:1" -Oe -0.84 -He 0.42 -frac 0.1 \
						   -outf 0-mc-tu-qv_autocoor_40ps.csv -seles "(around 4 (resname IAH) ) and type OW" \
						   -ncpu 8 -hann False

