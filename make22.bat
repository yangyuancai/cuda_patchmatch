rmdir /s /q build
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"  -DOpenCV_DIR="D:/ThridPartyLib/opencv/opencv_450ht/install/"
pause

