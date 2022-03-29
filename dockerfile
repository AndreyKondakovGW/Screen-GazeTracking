FROM python:3.8

WORKDIR /code

COPY ./ /code/

RUN apt-get update && apt-get upgrade -y
RUN apt-get install libopencv-highgui-dev -y
RUN apt -y install libusb-1.0-0-dev
RUN apt -y install ffmpeg libsm6 libxext6


# Install CMake GCC and other deps
RUN apt install build-essential -y
RUN apt install cmake -y
RUN apt-get install libx11-dev -y
RUN apt-get install xorg-dev libglu1-mesa-dev -y

# Install Realsense
RUN git clone https://github.com/IntelRealSense/librealsense.git
RUN cd librealsense
RUN cmake ./librealsense
RUN make uninstall && make clean && make && make install

#install Openvino
RUN echo "deb https://apt.repos.intel.com/openvino/2021 all main" | tee /etc/apt/sources.list.d/intel-openvino-2021.list ;
RUN wget https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021 ;
RUN apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2021  ;
RUN apt update ;
RUN apt-cache search openvino ;
RUN apt-cache search intel-openvino-runtime-ubuntu20 ;

RUN apt install -y intel-openvino-dev-ubuntu20-2021.4.582 ;

RUN echo 'source /opt/intel/openvino_2021/bin/setupvars.sh' >> /.bashrc

RUN pip install --no-cache-dir -r /code/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
# docker run -it --name NAME_FOR_CONTAINER -p 8000:8000 f206ad85320a
