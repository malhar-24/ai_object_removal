from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import os
from website.modulesforproject import main as mp

from django.conf import settings
def upload_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        fs = FileSystemStorage(location="media/")
        
        # fixed filename (always overwrite same file)
        filename = "uploaded_image.png"
        file_path = os.path.join(fs.location, filename)

        # delete old file if exists
        if os.path.exists(file_path):
            os.remove(file_path)

        # save new file with same name
        fs.save(filename, request.FILES["image"])

        return redirect("process", filename=filename)
    return render(request, "upload.html")

def process_image(request, filename):
    image_url = f"/media/{filename}"
    image_path = os.path.join(settings.MEDIA_ROOT, filename)  # real path

    if request.method == "POST":
        prompt = request.POST.get("prompt", "")
        output_path = mp.mask_pipeline(
            prompt, image_path, 15
        )
        overlay_path = os.path.join(settings.MEDIA_ROOT, "final_overlay.png")

        overlay_url = f"/media/{os.path.basename(overlay_path)}"
        request.session["overlay_url"] = overlay_url
        request.session["output_path"] = output_path
        request.session["original_image_url"] = image_url
        request.session["original_image_path"] = image_path  # ✅ store real path

        return redirect("overlay_page")

    return render(request, "process.html", {"image_url": image_url})



def overlay_page(request):
    overlay_url = request.session.get("overlay_url")
    if not overlay_url:
        return redirect("home")  # if accessed directly, redirect

    return render(request, "overlay.html", {"overlay_url": overlay_url})

from django.http import JsonResponse
import json



def store_points(request):
    if request.method == "POST":
        data = json.loads(request.body)

        # Extract and normalize points -> [[x, y], [x, y], ...]
        raw_points = data.get("points", [])
        points = [[p["x"], p["y"]] for p in raw_points if "x" in p and "y" in p]

        request.session["clicked_points"] = points
        print("Clicked points:", points)

        # Get overlay image path from session
        image_url = request.session.get("overlay_url")
        if not image_url:
            return JsonResponse({"status": "error", "message": "No image found in session"})

        # Convert URL (/media/foo.png) -> absolute filesystem path
        image_path = os.path.join(settings.BASE_DIR, image_url.lstrip("/"))

        # Run SAM + merge with existing mask
        mask_path, overlay_path = mp.ask_samv2_agent_with_point(
            image_path=image_path,
            points=points,
            grow_pixels=10,
            save_dir=settings.MEDIA_ROOT,
        )

        # Convert file paths -> URL paths for frontend
        mask_url = os.path.join("/media", os.path.basename(mask_path))
        overlay_url = os.path.join("/media", os.path.basename(overlay_path))

        # Store URLs in session
        request.session["overlay_url"] = overlay_url
        request.session["mask_url"] = mask_url

        return JsonResponse({
            "status": "ok",
            "points": points,   # now guaranteed to be [[x, y], ...]
            "overlay_url": overlay_url,
            "mask_url": mask_url,
        })

    return JsonResponse({"status": "error", "message": "Invalid request method"})






def options_page(request):
    overlay_url = request.session.get("overlay_url")
    mask_path = request.session.get("output_path")  
    img_path = request.session.get("original_image_path")  # ✅ use real path

    if not overlay_url or not mask_path or not img_path:
        return redirect("upload")

    result_url = None

    if request.method == "POST":
        action = request.POST.get("action")
        if(action=="square"):
            action="blur"
            style="square"
        elif(action=="bw_square"):
            action="blur"
            style="bw_square"
        elif(action=="blur"):
            style="normal"
        if(action=="remove"):
            style=None
        save_dir = settings.MEDIA_ROOT
        print(action,style)
        result = mp.perform_operation_with_mask(
            prompt="",
            image_path=img_path,   # ✅ now real path
            mask_pat=mask_path,
            action=action,
            save_dir=save_dir,
            blur_params={"style": style, "output_path": os.path.join(save_dir, "blur_result.png")},
            remove_params={"output_path": os.path.join(save_dir, "remove_result.png")}
        )
        print(action,style,result)

        if action == "blur":
            result_url = f"/media/{os.path.basename(result['blur'])}"
        elif action == "remove":
            result_url = f"/media/{os.path.basename(result['remove'])}"

    return render(request, "options.html", {
        "overlay_url": overlay_url,
        "result_url": result_url
    })
