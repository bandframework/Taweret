cd source/
sphinx-apidoc -f -E --implicit-namespaces -l -o . ../../src/Taweret
cd ..
make clean html
open build/html/Taweret.html
