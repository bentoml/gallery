function randomNumber(min, max) {
    return (Math.random() * (max - min) + min).toFixed(2);
}

$("#randomize").click(function () {
    $("#sepal_l").val(randomNumber(1, 5))
    $("#sepal_w").val(randomNumber(1, 5))
    $("#petal_l").val(randomNumber(1, 5))
    $("#petal_w").val(randomNumber(0, 2.5))
});


$("#predict").click(function () {
    let sample = [
        $("#sepal_l").val(),
        $("#sepal_w").val(),
        $("#petal_l").val(),
        $("#petal_w").val()
    ]
    predict(sample);
});

function predict(sample) {
    fetch("/test", {
        "headers": {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "method": "POST",
        // "mode": "cors",
        "body": JSON.stringify([sample])
    })
        .then(response => response.json())
        .then(res => {
            console.log(res);
            console.log(res[0]);
            if (res[0] == 0) {
                $("#setosa").fadeTo("slow", 1);
                $("#versicolor").fadeTo("slow", 0.33);
                $("#verginica").fadeTo("slow", 0.33);
            } else if (res[0] == 1) {
                $("#setosa").fadeTo("slow", 0.33);
                $("#versicolor").fadeTo("slow", 1);
                $("#verginica").fadeTo("slow", 0.33);
            } else {
                $("#setosa").fadeTo("slow", 0.33);
                $("#versicolor").fadeTo("slow", 0.33);
                $("#verginica").fadeTo("slow", 1);
            }
        })
        .catch(error => {
            console.log(error);
        });

}