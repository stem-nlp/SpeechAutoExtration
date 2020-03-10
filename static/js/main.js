let data_global = [];

$("#start-button").click(() => {
    let body = $("#input-body").val();
    let popup = $("#popup");

    if (body.length === 0) {
        popup.text("请输入新闻正文").show();
        return
    }
    popup.text("").hide();
    $("#start-button").attr('disabled', 'disabled');
    $("#button-spin").show();

    // http 请求
    $.ajax({
        url: "/api/model",
        type: "post",
        dataType: "json",
        data: {"body": body},
        success: (response) => {
            window.location.hash = "#start-button";
            if (response.code === 0) {
                renderOutput(response.data);
            } else {
                alert("服务器错误");
            }
            $("#start-button").removeAttr('disabled');
            $("#button-spin").hide();
        }
    });

});

// 渲染结果
function renderOutput(result) {
    let cnt = 0;
    $("#result-table tbody").empty();
    if (result.detail.length === 0) {
        alert("无结果");
        return;
    }
    for (let line of result.detail) {
        cnt += 1;
        let tableLine =
            `<tr>
                <td>${cnt}</td>
                <td>${line.speaker}</td>
                <td>${line.content}</td>
                <td class="${line.sentiment === 0 ? 'td-negative' : 'td-positive'}">
                    ${line.sentiment === 0 ? '负面' : '正面'}
                </td>
            </tr>`;
        $("#result-table tbody").append($(tableLine))
    }
    data_global = result.detail;
}


let btnList = document.querySelector(".btn-list");
let btnChart = document.querySelector(".btn-chart");
let newsList = document.querySelector(".news-list");
let newCharts = document.querySelector("#news-charts");

btnList.addEventListener("click", () => {

    newsList.style.display = "block";
    btnList.classList.add("active");
    newCharts.style.display = "none";
    btnChart.classList.remove("active");
});


// 图表按钮 -- 点击事件
btnChart.addEventListener("click", () => {

    newsList.style.display = "none";     // 列表视图隐藏
    newCharts.style.display = "block";   // 图表视图显示
    btnList.classList.remove("active");
    btnChart.classList.add("active");
    draw_echart(data_global);
});


function draw_echart(data) {
    let myChart = echarts.init(newCharts);    // 初始化echarts, 挂载至容器中

    // 饼状图参数
    myChart.setOption({
        title: {
            text: "新闻评论数据展示",
            left: "center"
        },
        tooltip: {
            trigger: 'item',
            formatter(params) {
                total_str = "";
                for(let i=0; i<data.length; i++){
                    if (data[i].sentiment === params.dataIndex){
                         total_str = total_str + data[i].content + `<br/>`;
                    }
                }
                return "<div style='font-size: x-small'>" + total_str + "</div>";
            }
        },
        legend: {
            orient: 'vertical',
            left: 'left',
            data: []
        },
        series: [
            {
                name: '言论',
                type: 'pie',
                radius: '60%',
                top: '-50',
                center: ['50%', '60%'],
                label: {
                    formatter: '{b}({d}%)'
                },
                data: [],
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }
        ]
    });

    // 根据数据渲染饼状图
    let names = ["负面", "正面"];
    let brower = [];
    let positive = 0;
    let negative = 0;
    // 获取原数据中属性及属性值
    data.forEach(item => {
        item.sentiment === 0 ? negative++ : positive++
    });
    brower.push({name: "负面", value: negative});
    brower.push({name: "正面", value: positive});

    // 更改饼状图参数
    myChart.setOption({
        legend: {
            data: names
        },
        series: [{
            data: brower
        }]
    });

    // // 重新获取鼠标滑过事件，浮动框渲染属性值
    // myChart.on("mouseover", function (params) {
    //
    //     // data.map((item, index) => {
    //     //     if (params.dataIndex === index) {
    //     //
    //     //         // params.seriesName = item.content;
    //     //         // return params.seriesName;
    //     //     }
    //     // });
    //     params.value = total_str;
    //     params.seriesName = item.content;
    //     // myChart.setOption({
    //     //     series: [{
    //     //         name: params.seriesName
    //     //     }]
    //     // });
    // });

}

