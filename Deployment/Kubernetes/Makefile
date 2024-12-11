VERSION=v1
DOCKERUSER=dishagundecha99
PROJECT=backend-service
build:
	mvn clean install -Dmaven.test.skip=true
	docker build -f Dockerfile -t ${PROJECT} --platform=linux/arm64 .

push:
	docker tag ${PROJECT} $(DOCKERUSER)/${PROJECT}:$(VERSION)
	docker push $(DOCKERUSER)/${PROJECT}:$(VERSION)
	docker tag ${PROJECT} $(DOCKERUSER)/${PROJECT}:latest
	docker push $(DOCKERUSER)/${PROJECT}:latest

deploy:
	kubectl apply -f ./kubernetes -n quantaas

cleanup:
	kubectl delete -f ./kubernetes -n quantaas

logs:
	kubectl logs -l app=backend-service -f

deploy-gke:
	kubectl apply -f ./kubernetes
	kubectl apply -f ./kubernetes-gke

cleanup-gke:
	kubectl delete -f ./kubernetes
	kubectl delete -f ./kubernetes-gke

deploy-minio:
	helm repo add bitnami https://charts.bitnami.com/bitnami
	helm install -f kubernetes-minio/minio-config.yaml -n minio-ns --create-namespace minio-proj bitnami/minio
	kubectl create -f kubernetes-minio/minio-external-service.yaml