while read -r yyyymm; do
  y=${yyyymm:0:4}; m=${yyyymm:4:2}
  wget -4 --no-http-keep-alive --show-progress -c \
    -O "ersst.v5.${y}${m}.nc" \
    "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ersst.v5.${y}${m}.nc"
  sleep 1
done < failed_yyyymm.txt
