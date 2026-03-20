from types import SimpleNamespace
from rest_framework.test import APIRequestFactory
from rest_framework.exceptions import PermissionDenied
from django.http import HttpResponse
import core.views as views
import pytest


def test_delete_existing_prediction_returns_early_without_url(monkeypatch):
    img = SimpleNamespace(predicted_url=None)

    monkeypatch.setenv("BUCKET_NAME", "my-bucket")
    called = {"delete": False}

    monkeypatch.setattr(views, "delete_file_from_bucket", lambda *args: called.update(delete=True))

    views._delete_existing_prediction(img)

    assert called["delete"] is False

def test_delete_existing_prediction_clears_fields_when_deleted(monkeypatch):
    class FakeImg:
        def __init__(self):
            self.predicted_url = "https://example.com/pred.jpg"
            self.predicted_at = "now"
            self.saved = False
            self.update_fields = None

        def save(self, update_fields=None):
            self.saved = True
            self.update_fields = update_fields

    img = FakeImg()

    monkeypatch.setenv("BUCKET_NAME", "my-bucket")
    monkeypatch.delenv("SKIP_CLOUD_UPLOAD", raising=False)
    monkeypatch.setattr(views, "delete_file_from_bucket", lambda url, bucket: True)

    views._delete_existing_prediction(img)

    assert img.predicted_url is None
    assert img.predicted_at is None
    assert img.saved is True
    assert img.update_fields == ["predicted_url", "predicted_at"]

def test_delete_existing_prediction_handles_delete_exception(monkeypatch):
    class FakeImg:
        def __init__(self):
            self.predicted_url = "https://example.com/pred.jpg"
            self.predicted_at = "now"
            self.saved = False

        def save(self, update_fields=None):
            self.saved = True

    img = FakeImg()

    monkeypatch.setenv("BUCKET_NAME", "my-bucket")
    monkeypatch.delenv("SKIP_CLOUD_UPLOAD", raising=False)

    def boom(url, bucket):
        raise Exception("bucket error")

    monkeypatch.setattr(views, "delete_file_from_bucket", boom)

    views._delete_existing_prediction(img)

    assert img.predicted_url == "https://example.com/pred.jpg"
    assert img.predicted_at == "now"
    assert img.saved is False

def test_run_prediction_for_image_success(monkeypatch, tmp_path):
    import core.views as views
    import core.services as services

    class FakeImg:
        def __init__(self):
            self.id = 1
            self.image_url = "img.jpg"
            self.predicted_url = None
            self.predicted_at = None
            self.detections = None
            self.saved = False
            self.update_fields = None

        def save(self, update_fields=None):
            self.saved = True
            self.update_fields = update_fields

    img = FakeImg()

    monkeypatch.setattr(views, "_delete_existing_prediction", lambda x: None)

    class FakeService:
        @staticmethod
        def predict(**kwargs):
            fake_path = tmp_path / "pred.jpg"
            fake_path.write_text("fake")
            return {"det": 1}, str(fake_path)

    monkeypatch.setattr(services, "RFDETRService", FakeService)

    monkeypatch.setattr(
        views,
        "upload_local_file_to_bucket",
        lambda path, bucket_name: "uploaded_url",
    )

    monkeypatch.setenv("BUCKET_NAME", "bucket")

    removed = {"called": False}
    monkeypatch.setattr(views.os, "remove", lambda path: removed.update(called=True))

    result = views._run_prediction_for_image(img, "normal", 0.5, 256)

    assert result["image_id"] == 1
    assert result["original_image"] == "img.jpg"
    assert result["predicted_image"] == "uploaded_url"
    assert result["detections"] == {"det": 1}

    assert img.predicted_url == "uploaded_url"
    assert img.detections == {"det": 1}
    assert img.saved is True
    assert img.update_fields == ["predicted_url", "predicted_at", "detections"]
    assert removed["called"] is True


def test_run_prediction_for_image_cleanup_error(monkeypatch):
    import core.views as views
    import core.services as services

    class FakeImg:
        def __init__(self):
            self.id = 1
            self.image_url = "img.jpg"
            self.predicted_url = None
            self.predicted_at = None
            self.detections = None

        def save(self, update_fields=None):
            self.update_fields = update_fields

    img = FakeImg()

    monkeypatch.setattr(views, "_delete_existing_prediction", lambda x: None)

    class FakeService:
        @staticmethod
        def predict(**kwargs):
            return {"det": 1}, "fake_path.jpg"

    monkeypatch.setattr(services, "RFDETRService", FakeService)
    monkeypatch.setattr(views, "upload_local_file_to_bucket", lambda *args, **kwargs: "url")
    monkeypatch.setenv("BUCKET_NAME", "bucket")
    monkeypatch.setattr(views.os.path, "exists", lambda path: True)

    def boom(path):
        raise Exception("delete failed")

    monkeypatch.setattr(views.os, "remove", boom)

    result = views._run_prediction_for_image(img, "normal", 0.5, 256)

    assert result["predicted_image"] == "url"
    assert result["detections"] == {"det": 1}


def test_sign_in_default_url(monkeypatch):
    import core.views as views

    class FakeRequest:
        pass

    captured = {}

    def fake_render(request, template, context):
        captured["template"] = template
        captured["context"] = context
        return "response"

    monkeypatch.setattr(views, "render", fake_render)
    monkeypatch.delenv("SIGN_IN_URL", raising=False)

    response = views.sign_in(FakeRequest())

    assert captured["template"] == "backend/sign_in.html"
    assert captured["context"]["sign_in_url"] == "http://localhost:8000/api/google/auth/"

def test_sign_in_adds_https(monkeypatch):
    import core.views as views

    class FakeRequest:
        pass

    captured = {}

    def fake_render(request, template, context):
        captured["context"] = context
        return "response"

    monkeypatch.setattr(views, "render", fake_render)
    monkeypatch.setenv("SIGN_IN_URL", "example.com/login")

    views.sign_in(FakeRequest())

    assert captured["context"]["sign_in_url"] == "https://example.com/login"

def test_sign_in_keeps_valid_url(monkeypatch):
    import core.views as views

    class FakeRequest:
        pass

    captured = {}

    def fake_render(request, template, context):
        captured["context"] = context
        return "response"

    monkeypatch.setattr(views, "render", fake_render)
    monkeypatch.setenv("SIGN_IN_URL", "https://valid.com")

    views.sign_in(FakeRequest())

    assert captured["context"]["sign_in_url"] == "https://valid.com"

def test_google_login_modal(monkeypatch):
    import core.views as views

    class FakeRequest:
        pass

    captured = {}

    def fake_render(request, template):
        captured["template"] = template
        return "response"

    monkeypatch.setattr(views, "render", fake_render)

    views.google_login_modal(FakeRequest())

    assert captured["template"] == "backend/modals/google_login_modal.html"

def test_auth_receive_missing_credential():
    factory = APIRequestFactory()
    request = factory.post("/google/auth", {}, format="json")

    response = views.auth_receive(request)

    assert response.status_code == 400
    assert response.data == {"error": "Missing credential"}

def test_auth_receive_invalid_token(monkeypatch):
    factory = APIRequestFactory()
    request = factory.post("/google/auth", {"credential": "bad-token"}, format="json")

    def fake_verify(token, req, client_id):
        raise ValueError("invalid")

    monkeypatch.setattr(views.id_token, "verify_oauth2_token", fake_verify)

    response = views.auth_receive(request)

    assert response.status_code == 400
    assert response.data == {"error": "Invalid token."}

def test_auth_receive_success_sets_cookies_and_redirects(monkeypatch):
    factory = APIRequestFactory()
    request = factory.post("/google/auth", {"credential": "good-token"}, format="json")
    request.META["HTTP_HOST"] = "localhost:8000"

    monkeypatch.setattr(
        views.id_token,
        "verify_oauth2_token",
        lambda token, req, client_id: {
            "email": "test@example.com",
            "name": "Test User",
        },
    )

    class FakeUser:
        email = "test@example.com"

    class FakeManager:
        @staticmethod
        def get_or_create(email, defaults):
            return FakeUser(), True

    monkeypatch.setattr(views.User, "objects", FakeManager())

    class FakeRefresh:
        access_token = "access123"

        def __str__(self):
            return "refresh123"

    class FakeRefreshToken:
        @staticmethod
        def for_user(user):
            return FakeRefresh()

    monkeypatch.setattr(views, "RefreshToken", FakeRefreshToken)
    monkeypatch.delenv("FRONTEND_BASE_URL", raising=False)
    monkeypatch.delenv("FRONTEND_CUSTOMERS_URL", raising=False)
    monkeypatch.delenv("COOKIE_DOMAIN", raising=False)

    response = views.auth_receive(request)

    assert response.status_code == 302
    assert response["Location"] == "http://localhost:8000/customers"

    access_cookie = response.cookies["access"]
    refresh_cookie = response.cookies["refresh"]

    assert access_cookie.value == "access123"
    assert refresh_cookie.value == "refresh123"
    assert access_cookie["path"] == "/"
    assert refresh_cookie["path"] == "/"

def test_auth_receive_sets_cookie_domain_when_host_matches(monkeypatch):
    factory = APIRequestFactory()
    request = factory.post("/google/auth", {"credential": "good-token"}, format="json")
    request.get_host = lambda: "app.example.com"

    monkeypatch.setattr(
        views.id_token,
        "verify_oauth2_token",
        lambda token, req, client_id: {
            "email": "test@example.com",
            "name": "Test User",
        },
    )

    class FakeUser:
        email = "test@example.com"

    class FakeManager:
        @staticmethod
        def get_or_create(email, defaults):
            return FakeUser(), False

    monkeypatch.setattr(views.User, "objects", FakeManager())

    class FakeRefresh:
        access_token = "access123"

        def __str__(self):
            return "refresh123"

    class FakeRefreshToken:
        @staticmethod
        def for_user(user):
            return FakeRefresh()

    monkeypatch.setattr(views, "RefreshToken", FakeRefreshToken)
    monkeypatch.setenv("COOKIE_DOMAIN", "example.com")
    monkeypatch.setenv("FRONTEND_BASE_URL", "https://frontend.example.com")
    monkeypatch.delenv("FRONTEND_CUSTOMERS_URL", raising=False)

    response = views.auth_receive(request)

    assert response.status_code == 302
    assert response["Location"] == "https://frontend.example.com/customers"
    assert response.cookies["access"]["domain"] == "example.com"
    assert response.cookies["refresh"]["domain"] == "example.com"

def test_sign_out_deletes_cookies_without_domain(monkeypatch):
    factory = APIRequestFactory()
    request = factory.get("/signout")
    request.get_host = lambda: "localhost:8000"

    monkeypatch.delenv("COOKIE_DOMAIN", raising=False)

    response = views.sign_out(request)

    assert response.status_code == 302
    assert "access" in response.cookies
    assert "refresh" in response.cookies
    assert response.cookies["access"]["path"] == "/"
    assert response.cookies["refresh"]["path"] == "/"

def test_sign_out_deletes_domain_cookies_when_host_matches(monkeypatch):
    factory = APIRequestFactory()
    request = factory.get("/signout")
    request.get_host = lambda: "app.example.com"

    monkeypatch.setenv("COOKIE_DOMAIN", "example.com")

    response = views.sign_out(request)

    assert response.status_code == 302
    assert response.cookies["access"]["domain"] == "example.com"
    assert response.cookies["refresh"]["domain"] == "example.com"


def test_index_returns_not_authenticated_when_no_debug_user():
    factory = APIRequestFactory()
    request = factory.get("/")

    request.session = {}

    response = views.index(request)

    assert response.status_code == 200
    assert response.data == {
        "message": "User not authenticated or pass token of jwt"
    }


def test_index_returns_debug_user_when_present():
    factory = APIRequestFactory()
    request = factory.get("/")

    request.session = {"debug_user": "jeffery"}

    response = views.index(request)

    assert response.status_code == 200
    assert response.data == {
        "message": "Hello from the backend! with session test",
        "user": "jeffery",
    }

def test_is_agent_owner_customer_match():
    import core.views as views

    perm = views.IsAgentOwner()

    class Obj:
        agent = "user1"

    class Request:
        user = "user1"

    assert perm.has_object_permission(Request(), None, Obj()) is True

def test_is_agent_owner_customer_no_match():
    import core.views as views

    perm = views.IsAgentOwner()

    class Obj:
        agent = "user1"

    class Request:
        user = "user2"

    assert perm.has_object_permission(Request(), None, Obj()) is False

def test_is_agent_owner_house():
    import core.views as views

    perm = views.IsAgentOwner()

    class Customer:
        agent = "user1"

    class Obj:
        customer = Customer()

    class Request:
        user = "user1"

    assert perm.has_object_permission(Request(), None, Obj()) is True

def test_is_agent_owner_house_image():
    import core.views as views

    perm = views.IsAgentOwner()

    class Customer:
        agent = "user1"

    class House:
        customer = Customer()

    class Obj:
        house = House()

    class Request:
        user = "user1"

    assert perm.has_object_permission(Request(), None, Obj()) is True

def test_is_agent_owner_fallback_false():
    import core.views as views

    perm = views.IsAgentOwner()

    class Obj:
        pass  # no agent/customer/house

    class Request:
        user = "user1"

    assert perm.has_object_permission(Request(), None, Obj()) is False

def test_house_viewset_get_queryset_filters_by_request_user():
    import core.views as views

    class FakeQueryset:
        def __init__(self):
            self.kwargs = None

        def filter(self, **kwargs):
            self.kwargs = kwargs
            return "filtered_result"

    viewset = views.HouseViewSet()
    viewset.queryset = FakeQueryset()

    class Request:
        user = "agent1"

    viewset.request = Request()

    result = viewset.get_queryset()

    assert result == "filtered_result"
    assert viewset.queryset.kwargs == {"customer__agent": "agent1"}

def test_house_viewset_perform_create_saves_when_customer_owned():
    import core.views as views

    viewset = views.HouseViewSet()

    class Request:
        user = "agent1"

    viewset.request = Request()

    class Customer:
        agent = "agent1"

    class FakeSerializer:
        validated_data = {"customer": Customer()}

        def __init__(self):
            self.saved = False

        def save(self):
            self.saved = True

    serializer = FakeSerializer()

    viewset.perform_create(serializer)

    assert serializer.saved is True

def test_house_viewset_perform_create_raises_when_customer_not_owned():
    import core.views as views

    viewset = views.HouseViewSet()

    class Request:
        user = "agent1"

    viewset.request = Request()

    class Customer:
        agent = "other-agent"

    class FakeSerializer:
        validated_data = {"customer": Customer()}

        def save(self):
            raise AssertionError("save should not be called")

    with pytest.raises(PermissionDenied, match="You do not own this customer."):
        viewset.perform_create(FakeSerializer())

def test_house_viewset_destroy_deletes_images_and_returns_204(monkeypatch):
    import core.views as views

    deleted = []

    monkeypatch.setattr(views, "delete_file_from_bucket", lambda url, bucket: deleted.append((url, bucket)))
    monkeypatch.setenv("BUCKET_NAME", "my-bucket")

    class Img:
        def __init__(self, image_url):
            self.image_url = image_url

    class ImagesManager:
        def all(self):
            return [Img("url1"), Img(None), Img("url2")]

    class Instance:
        images = ImagesManager()

    viewset = views.HouseViewSet()

    instance = Instance()
    monkeypatch.setattr(viewset, "get_object", lambda: instance)

    destroyed = {"called": False}
    monkeypatch.setattr(viewset, "perform_destroy", lambda obj: destroyed.update(called=obj is instance))

    response = viewset.destroy(request=None)

    assert response.status_code == 204
    assert destroyed["called"] is True
    assert deleted == [("url1", "my-bucket"), ("url2", "my-bucket")]

def test_house_viewset_destroy_handles_bucket_delete_exception(monkeypatch):
    import core.views as views

    monkeypatch.setenv("BUCKET_NAME", "my-bucket")

    class Img:
        image_url = "url1"

    class ImagesManager:
        def all(self):
            return [Img()]

    class Instance:
        images = ImagesManager()

    viewset = views.HouseViewSet()
    instance = Instance()

    monkeypatch.setattr(viewset, "get_object", lambda: instance)

    destroyed = {"called": False}
    monkeypatch.setattr(viewset, "perform_destroy", lambda obj: destroyed.update(called=True))

    def boom(url, bucket):
        raise Exception("delete failed")

    monkeypatch.setattr(views, "delete_file_from_bucket", boom)

    response = viewset.destroy(request=None)

    assert response.status_code == 204
    assert destroyed["called"] is True

def test_houseimage_viewset_get_queryset_filters_by_request_user(monkeypatch):
    import core.views as views

    captured = {}

    class FakeManager:
        def filter(self, **kwargs):
            captured["kwargs"] = kwargs
            return "filtered_images"

    monkeypatch.setattr(views.HouseImage, "objects", FakeManager())

    viewset = views.HouseImageViewSet()

    class Request:
        user = "agent1"

    viewset.request = Request()

    result = viewset.get_queryset()

    assert result == "filtered_images"
    assert captured["kwargs"] == {"house__customer__agent": "agent1"}

def test_houseimage_viewset_destroy_deletes_bucket_file_and_returns_204(monkeypatch):
    import core.views as views

    monkeypatch.setenv("BUCKET_NAME", "bucket1")

    deleted = []

    monkeypatch.setattr(
        views,
        "delete_file_from_bucket",
        lambda url, bucket: deleted.append((url, bucket)),
    )

    class Instance:
        image_url = "https://example.com/img.jpg"

    viewset = views.HouseImageViewSet()
    instance = Instance()

    monkeypatch.setattr(viewset, "get_object", lambda: instance)

    destroyed = {"called": False}
    monkeypatch.setattr(
        viewset,
        "perform_destroy",
        lambda obj: destroyed.update(called=(obj is instance)),
    )

    response = viewset.destroy(request=None)

    assert response.status_code == 204
    assert destroyed["called"] is True
    assert deleted == [("https://example.com/img.jpg", "bucket1")]

def test_houseimage_viewset_destroy_handles_delete_exception(monkeypatch):
    import core.views as views

    monkeypatch.setenv("BUCKET_NAME", "bucket1")

    def boom(url, bucket):
        raise Exception("delete failed")

    monkeypatch.setattr(views, "delete_file_from_bucket", boom)

    class Instance:
        image_url = "https://example.com/img.jpg"

    viewset = views.HouseImageViewSet()
    instance = Instance()

    monkeypatch.setattr(viewset, "get_object", lambda: instance)

    destroyed = {"called": False}
    monkeypatch.setattr(
        viewset,
        "perform_destroy",
        lambda obj: destroyed.update(called=True),
    )

    response = viewset.destroy(request=None)

    assert response.status_code == 204
    assert destroyed["called"] is True

def test_houseimage_viewset_perform_create_raises_when_no_file():
    import core.views as views

    viewset = views.HouseImageViewSet()

    class Files:
        @staticmethod
        def get(name):
            return None

    class Request:
        FILES = Files()

    viewset.request = Request()

    class FakeSerializer:
        def save(self, **kwargs):
            raise AssertionError("save should not be called")

    with pytest.raises(Exception, match="No file uploaded"):
        viewset.perform_create(FakeSerializer())


def test_houseimage_viewset_perform_create_raises_when_upload_fails(monkeypatch):
    import core.views as views

    viewset = views.HouseImageViewSet()

    fake_file = object()

    class Files:
        @staticmethod
        def get(name):
            return fake_file

    class Request:
        FILES = Files()

    viewset.request = Request()

    monkeypatch.setattr(views, "upload_file_to_bucket", lambda file_obj, bucket_name: None)

    class FakeSerializer:
        def save(self, **kwargs):
            raise AssertionError("save should not be called")

    with pytest.raises(Exception, match="Failed to upload image"):
        viewset.perform_create(FakeSerializer())

def test_houseimage_viewset_perform_create_saves_image_url(monkeypatch):
    import core.views as views

    viewset = views.HouseImageViewSet()

    fake_file = object()

    class Files:
        @staticmethod
        def get(name):
            return fake_file

    class Request:
        FILES = Files()

    viewset.request = Request()

    monkeypatch.setenv("BUCKET_NAME", "bucket1")
    monkeypatch.setattr(
        views,
        "upload_file_to_bucket",
        lambda file_obj, bucket_name: "https://example.com/uploaded.jpg",
    )

    class FakeSerializer:
        def __init__(self):
            self.saved_kwargs = None

        def save(self, **kwargs):
            self.saved_kwargs = kwargs

    serializer = FakeSerializer()

    viewset.perform_create(serializer)

    assert serializer.saved_kwargs == {
        "image_url": "https://example.com/uploaded.jpg"
    }

def test_redirect_404_redirects_to_login(monkeypatch):
    import core.views as views

    captured = {}

    def fake_redirect(name):
        captured["name"] = name
        return "redirect-response"

    monkeypatch.setattr(views, "redirect", fake_redirect)

    response = views.redirect_404(request=None, exception=None)

    assert response == "redirect-response"
    assert captured["name"] == "login"



def test_image_prediction_returns_404_when_image_not_found(monkeypatch):
    factory = APIRequestFactory()
    request = factory.post("/image-prediction", {}, format="json")

    def boom(**kwargs):
        raise views.HouseImage.DoesNotExist()

    monkeypatch.setattr(views.HouseImage.objects, "get", boom)

    response = views.image_prediction(request, image_id=123)

    assert response.status_code == 404
    assert response.data == {"error": "Image not found"}


def test_image_prediction_returns_result(monkeypatch):
    factory = APIRequestFactory()
    request = factory.post(
        "/image-prediction",
        {"mode": "tiled", "threshold": 0.7, "tile_size": 640},
        format="json",
    )

    class FakeImg:
        id = 5
        image_url = "https://example.com/img.jpg"

    monkeypatch.setattr(views.HouseImage.objects, "get", lambda **kwargs: FakeImg())
    monkeypatch.setattr(
        views,
        "_run_prediction_for_image",
        lambda img, mode, threshold, tile_size: {
            "image_id": img.id,
            "original_image": img.image_url,
            "predicted_image": "https://example.com/pred.jpg",
            "detections": {"scores": [0.9]},
        },
    )

    response = views.image_prediction(request, image_id=5)

    assert response.status_code == 200
    assert response.data["image_id"] == 5
    assert response.data["original_image"] == "https://example.com/img.jpg"
    assert response.data["predicted_image"] == "https://example.com/pred.jpg"

def test_image_prediction_returns_500_when_prediction_fails(monkeypatch):
    factory = APIRequestFactory()
    request = factory.post("/image-prediction", {}, format="json")

    class FakeImg:
        id = 7
        image_url = "https://example.com/img2.jpg"

    monkeypatch.setattr(views.HouseImage.objects, "get", lambda **kwargs: FakeImg())

    def boom(img, mode, threshold, tile_size):
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(views, "_run_prediction_for_image", boom)

    response = views.image_prediction(request, image_id=7)

    assert response.status_code == 500
    assert response.data["image_id"] == 7
    assert response.data["original_image"] == "https://example.com/img2.jpg"
    assert "RuntimeError: prediction failed" in response.data["error"]

def test_run_prediction_returns_404_when_house_not_found(monkeypatch):
    factory = APIRequestFactory()
    request = factory.post("/run-prediction", {}, format="json")

    class FakeManager:
        def prefetch_related(self, *args, **kwargs):
            return self

        def get(self, **kwargs):
            raise views.House.DoesNotExist()

    monkeypatch.setattr(views.House, "objects", FakeManager())

    response = views.run_prediction(request, house_id=1)

    assert response.status_code == 404
    assert response.data == {"error": "House not found"}

def test_run_prediction_returns_message_when_no_images(monkeypatch):
    factory = APIRequestFactory()
    request = factory.post("/run-prediction", {}, format="json")

    class FakeImages:
        def exists(self):
            return False

    class FakeHouse:
        images = FakeImages()

    class FakeManager:
        def prefetch_related(self, *args, **kwargs):
            return self

        def get(self, **kwargs):
            return FakeHouse()

    monkeypatch.setattr(views.House, "objects", FakeManager())

    response = views.run_prediction(request, house_id=2)

    assert response.status_code == 200
    assert response.data == {"message": "No images found for this house."}


def test_run_prediction_get_renders_template(monkeypatch):
    factory = APIRequestFactory()
    request = factory.get("/run-prediction")

    class FakeImages:
        def exists(self):
            return True

        def all(self):
            return ["img1", "img2"]

    class FakeHouse:
        images = FakeImages()

    class FakeManager:
        def prefetch_related(self, *args, **kwargs):
            return self

        def get(self, **kwargs):
            return FakeHouse()

    monkeypatch.setattr(views.House, "objects", FakeManager())

    captured = {}

    def fake_render(request, template, context):
        captured["template"] = template
        captured["context"] = context
        return HttpResponse("rendered")

    monkeypatch.setattr(views, "render", fake_render)

    response = views.run_prediction(request, house_id=3)

    assert response.status_code == 200
    assert captured["template"] == "backend/run_prediction.html"
    assert captured["context"]["house_id"] == 3
    assert captured["context"]["images"] == ["img1", "img2"]


def test_run_prediction_post_success_for_all_images(monkeypatch):
    factory = APIRequestFactory()
    request = factory.post(
        "/run-prediction",
        {"mode": "tiled", "threshold": 0.8, "tile_size": 640},
        format="json",
    )

    class Img:
        def __init__(self, id_, url):
            self.id = id_
            self.image_url = url

    class FakeImages:
        def exists(self):
            return True

        def all(self):
            return [
                Img(1, "img1.jpg"),
                Img(2, "img2.jpg"),
            ]

    class FakeHouse:
        images = FakeImages()

    class FakeManager:
        def prefetch_related(self, *args, **kwargs):
            return self

        def get(self, **kwargs):
            return FakeHouse()

    monkeypatch.setattr(views.House, "objects", FakeManager())

    monkeypatch.setattr(
        views,
        "_run_prediction_for_image",
        lambda img, mode, threshold, tile_size: {
            "image_id": img.id,
            "original_image": img.image_url,
            "predicted_image": f"pred-{img.id}.jpg",
            "detections": {"ok": True},
        },
    )

    response = views.run_prediction(request, house_id=4)

    assert response.status_code == 200
    assert response.data["house_id"] == 4
    assert response.data["total_images"] == 2
    assert len(response.data["results"]) == 2
    assert response.data["results"][0]["predicted_image"] == "pred-1.jpg"
    assert response.data["results"][1]["predicted_image"] == "pred-2.jpg"

def test_run_prediction_post_collects_errors_per_image(monkeypatch):
    factory = APIRequestFactory()
    request = factory.post("/run-prediction", {}, format="json")

    class Img:
        def __init__(self, id_, url):
            self.id = id_
            self.image_url = url

    imgs = [
        Img(1, "img1.jpg"),
        Img(2, "img2.jpg"),
    ]

    class FakeImages:
        def exists(self):
            return True

        def all(self):
            return imgs

    class FakeHouse:
        images = FakeImages()

    class FakeManager:
        def prefetch_related(self, *args, **kwargs):
            return self

        def get(self, **kwargs):
            return FakeHouse()

    monkeypatch.setattr(views.House, "objects", FakeManager())

    def fake_run(img, mode, threshold, tile_size):
        if img.id == 1:
            return {
                "image_id": img.id,
                "original_image": img.image_url,
                "predicted_image": "pred-1.jpg",
                "detections": {"ok": True},
            }
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(views, "_run_prediction_for_image", fake_run)

    response = views.run_prediction(request, house_id=5)

    assert response.status_code == 200
    assert response.data["house_id"] == 5
    assert response.data["total_images"] == 2
    assert response.data["results"][0]["image_id"] == 1
    assert response.data["results"][1]["image_id"] == 2
    assert response.data["results"][1]["original_image"] == "img2.jpg"
    assert "RuntimeError: prediction failed" in response.data["results"][1]["error"]