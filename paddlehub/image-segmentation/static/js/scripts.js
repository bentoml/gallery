function previewFile() {
    let preview = document.getElementById("preview");
    let file    = document.querySelector('input[type=file]').files[0];
    let reader  = new FileReader();
  
    if (file) {
      reader.readAsDataURL(file);
    } else {
      preview.src = "";
    }

    reader.onloadend = function () {
        preview.src = reader.result;
        $("#download > a").attr("href", reader.result);
    }
}

function segmentation() {
    $("#spinner").css("display", "inline-block");
    let file    = document.querySelector('input[type=file]').files[0];

    const formData = new FormData()
    formData.append('image', file)
    
    fetch("/test", {
        "headers": {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Access-Control-Allow-Origin": "*"
        },
        "method": "POST",
        "body": formData
    })
        .then(response => response.json())
        .then(res => {
            $("#spinner").css("display", "none");
            displayImage(res);
        })
        .catch(error => {
            console.log(error);
        });

}

function displayImage(imgArray) {
    let width = Number(imgArray[0].length);
    let height = Number(imgArray.length);

    // Create canvas
    let canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    let ctx = canvas.getContext('2d');

    // Flatten imgArray
    const flattenedRGBAValues = imgArray.flat(Infinity)

    // Convert imgArray to ImageData
    const imgData = new ImageData(Uint8ClampedArray.from(flattenedRGBAValues), width, height);
    ctx.putImageData(imgData, 0, 0);

    // Display image to output preview
    let outputPreview = $("img#output");
    outputPreview.width($("#img#preview").width());
    outputPreview.height($("#img#preview").height());
    outputPreview.attr('src', canvas.toDataURL());
    outputPreview.css("display", "inline-block");
    
    // Display download button
    $("#download > a").attr("href", canvas.toDataURL());
    $("#download").css("display", "inline-block");
}

function download() {
    $("#download > a")[0].click();
}