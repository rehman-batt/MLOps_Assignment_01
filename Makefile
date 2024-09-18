build:
	docker build --tag mera_pehla_container .
images:
	docker images
remove-img:
	docker rmi mera_pehla_container:latest
run:
	docker run -d -p 5000:5000 --name mera_pehla_container_running mera_pehla_container
stop:
	docker stop $(container_id)
id:
	docker ps -a
remove-cnt:
	docker container rm $(container_id)