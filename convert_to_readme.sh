file_and_folder=$1
filename=`basename $1`
foldername=`dirname $1`
mkdir $foldername/temp
convert $1 $foldername/temp/output.png
ls -1 $foldername/temp/ | sort -n -k1.8 | sed 's/^/![](/g' | sed 's/$/)/g' > $foldername/README.md
