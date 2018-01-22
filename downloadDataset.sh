#!/usr/bin/env bash

echo "Downloading Training Dataset: German Traffic Signs"
url="https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
hashSum="d1f6275a17123ce3fab090105bf222af"
repo_root="$(dirname $0)"

download_path="/trainingData/"
path="${repo_root}${download_path}"
filename="traffic-signs-data.zip"

# does file exist?
if [ -f "${path}/traffic-signs-data.zip" ]
then
    checksum=`md5sum "${path}/${filename}" | awk '{ print $1 }'`
    if ! [[ "${hashSum}" == "${checksum}" ]]
        then
            echo "Download Corrupt! Retrying"
            exit 0
            rm "${path}/${filename}"
            wget -O "${path}/${filename}" "$url"
        else
            echo "File already exists!"
        fi
else
	mkdir -p ${path}
	wget -O "${path}/${filename}" "$url"
fi

unzip "${path}/${filename}" -d "$path"
exit 0

