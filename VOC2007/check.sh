#!/bin/bash

XML_PATH="/Users/manlu/Desktop/VOC2007/Annotations/"
JPEG_PATH="/Users/manlu/Desktop/VOC2007/JPEGImages/"

echo "Checking that all XML files have an equivalent JPG file ..."

for f in $XML_PATH*.xml;
do
  fn=$(basename $f);
  fileName=$(echo ${fn%.*});

  if [ ! -f $JPEG_PATH$fileName".jpg" ]
  then
    echo "File: "$fn" does not match any "$fileName".jpg file";
  fi
done

echo "Checking that all JPG files have an equivalent XML file ..."

for f in $JPEG_PATH/*.jpg;
do
  fn=$(basename $f);
  fileName=$(echo ${fn%.*});

  if [ ! -f $XML_PATH$fileName".xml" ]
  then
    echo "File: "$fn" does not match any "$fileName".xml file";
  fi
done
