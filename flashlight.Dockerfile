FROM flml/flashlight:cuda-dc12a10

RUN pip3 install torch==1.2.0 packaging==19.1
RUN cd /root/flashlight/bindings/python && python3 setup.py install
