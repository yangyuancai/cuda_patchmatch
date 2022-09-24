rmdir /s /q build
mkdir build
cd build
cmake .. -G "Visual Studio 15 2017 Win64"  -DOpenCV_DIR="D:/ThridPartyLib/opencv/opencv_450AllInOne/install/"
pause

