cd source/
sphinx-apidoc -f -E --implicit-namespaces -l -o . ../../Taweret
cd ..
make clean html
open build/html/Taweret.html
