function fillOutput(result){
}

$("#start-button").click(()=>{
    let body = $("#input-body").val();
    let popup = $("#popup");

    if (body.length === 0){
        popup.text("请输入新闻正文").show();
        return
    }
    popup.text("").hide();

    // http 请求
    $.ajax({
        url:"/api/model",
        type: "post",
        dataType: "json",
        data: {"body":body},
        success: (response)=>{
            window.location.hash = "#start-button";
            if(response.code === 0){
                renderOutput(response.data);
            }else {
                alert("服务器错误");
            }
        }
    });

});

// 渲染结果
function renderOutput(result){
    let cnt=0;
    $("#result-table tbody").empty();
    if(result.detail.length === 0){
        alert("无结果");
    }
    for (let line of result.detail){
        cnt += 1;
        let tableLine = `<tr>
                            <td>${cnt}</td>
                            <td>${line.speaker}</td>
                            <td>${line.content}</td>
                        </tr>`;
        $("#result-table tbody").append($(tableLine))
    }
}
