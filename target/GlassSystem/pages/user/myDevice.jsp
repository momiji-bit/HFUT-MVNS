<%@ page contentType="text/html;charset=UTF-8" language="java" isELIgnored="false" %>
<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c"%>
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>layui</title>
    <meta name="renderer" content="webkit">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
    <link rel="stylesheet" href="../../statics/layui/lib/layui-v2.5.5/css/layui.css" media="all">
    <link rel="stylesheet" href="../../statics/layui/css/public.css" media="all">
</head>
<body>
<div class="layuimini-container">
    <div class="layuimini-main">

        <ins class="adsbygoogle" style="display:inline-block;width:970px;height:90px" data-ad-client="ca-pub-6111334333458862" data-ad-slot="3820120620"></ins>

        <fieldset class="layui-elem-field layui-field-title" style="margin-top: 50px;">
            <legend>设备功能开关</legend>
        </fieldset>



        <form class="layui-form" action="" lay-filter="example">

            <div class="layui-form-item">
                <label class="layui-form-label">我的设备ID</label>
                <div class="layui-input-block">
                    <select lay-filter="device" name="device" lay-verify="device" >
                            <option value="null">选择我的设备</option>
                            <option value="${myDeviceId}">${myDeviceId}</option>
                    </select>
                </div>
            </div>



            <div class="layui-form-item">
                <label class="layui-form-label">碰撞检测</label>
                <div class="layui-input-block">
                    <input type="checkbox" name="collide" lay-skin="switch" lay-text="开启|关闭">
                </div>
            </div>
            <div class="layui-form-item">
                <label class="layui-form-label">道路边缘检测</label>
                <div class="layui-input-block">
                    <input type="checkbox" name="edge" lay-skin="switch" lay-text="开启|关闭">
                </div>
            </div>
            <div class="layui-form-item">
                <label class="layui-form-label">手势识别</label>
                <div class="layui-input-block">

                    <input type="checkbox" name="gesture" lay-skin="switch" lay-text="开启|关闭">
                </div>
            </div>
            <div class="layui-form-item">
                <label class="layui-form-label">语音播报</label>
                <div class="layui-input-block">
                    <input type="checkbox" name="broadcast" lay-skin="switch" lay-text="开启|关闭">

                </div>
            </div>
            <div class="layui-form-item">
                <label class="layui-form-label">设备导航</label>
                <div class="layui-input-block">
                    <input type="checkbox" name="navigation" lay-skin="switch" lay-text="开启|关闭">

                </div>
            </div>

            <div class="layui-form-item">
                <div class="layui-input-block">
                    <button class="layui-btn" lay-submit="" lay-filter="demo1">立即提交</button>
                </div>
            </div>
        </form>

    </div>
</div>

<script src="../../statics/layui/lib/layui-v2.5.5/layui.js" charset="utf-8"></script>
<!-- 注意：如果你直接复制所有代码到本地，上述js路径需要改成你本地的 -->
<script>
    layui.use(['form', 'layedit', 'laydate'], function () {
        var form = layui.form
            , layer = layui.layer
            , $=layui.jquery;



        form.verify({
            device: function(value, item){ //value：表单的值、item：表单的DOM对象
                //如果不想自动弹出默认提示框，可以直接返回 true，这时你可以通过其他任意方式提示（v2.5.7 新增）
                if(value == "null"){
                    return '请至少选择一个设备';
                }
            }

        });


        form.on('select(device)',function (data) {
            var deviceId=data.value;
            if(deviceId!="null"){
                $.ajax({
                    type:'POST',
                    url: '/findDeviceByDeviceId.do',
                    data:{
                        "deviceId":deviceId
                    },
                    dataType:"json",
                    success: function (data) {
                        form.val('example', {
                            "collide": data.collide
                            , "edge": data.edge
                            , "gesture": data.gesture //复选框选中状态
                            , "broadcast": data.broadcast //开关状态
                            , "navigation": data.navigation

                        });
                        layer.alert("成功查询到该设备的状态")
                    },
                    error:function () {
                        layer.alert("请求失败")
                    }

                })
            }   else  {
                form.val('example', {

                    "collide": false
                    , "edge": false
                    , "gesture": false //复选框选中状态
                    , "broadcast": false //开关状态
                    , "navigation": false

                })

            }


        })
        //监听提交
        form.on('submit(demo1)', function (data) {
            data.field.collide = data.field.collide=="on"? "1":"0";
            data.field.edge = data.field.edge=="on"? "1":"0";
            data.field.gesture=data.field.gesture=="on"? "1":"0";
            data.field.broadcast=data.field.broadcast=="on"? "1":"0";
            data.field.navigation=data.field.navigation=="on"? "1":"0";

            $.ajax({
                type:'POST',
                url:'/updateStatus.do',
                data:data.field,
                dataType: 'json',
                success:function (data){
                    var info= data.success?"设备状态更改成功":"设备状态更改失败,请检查设备ID是否正确";
                    layer.alert(JSON.stringify(info), {
                        title: '提示'
                    })
                },
                error:function (data) {
                    var info= "设备状态更改失败，请检查网络是否正常";
                    layer.alert(JSON.stringify(info), {
                        title: '提示'
                    })
                }
            })
            return false;
        });


        form.render();




    });
</script>

</body>
</html>