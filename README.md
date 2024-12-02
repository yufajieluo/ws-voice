# ws-voice

## install
	1. Install python 3.11.10
	wget https://www.python.org/ftp/python/3.11.10/Python-3.11.10.tgz
	tar -zxf Python-3.11.10.tgz
	cd Python-3.11.10/
	./configure --enable-optimizations --prefix=/usr/local
	make -j $(nproc)
	sudo make altinstall
	
	2. Configure venv
	cd ${work_folder}
	python3.11 -m venv venv
	source venv/bin/activate
	
http://wsnote.oss-cn-beijing.aliyuncs.com/softwares/chattts-models/asset.tar.gz

