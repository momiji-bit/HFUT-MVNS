<%@ page contentType="text/html;charset=UTF-8" language="java" isELIgnored="false" %>
<%
    String basePath = request.getScheme() + "://" + request.getServerName() + ":" + request.getServerPort() + request.getContextPath() + "/";
%>
<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c"%>
<!DOCTYPE html>
<html>
<head>
    <base href="<%=basePath%>">
    <meta charset="utf-8">
    <title>layui</title>
    <meta name="renderer" content="webkit">
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
    <link rel="stylesheet" href="../../statics/layui/lib/layui-v2.5.5/css/layui.css" media="all">
    <link rel="stylesheet" href="../../statics/layui/css/public.css" media="all">
</head>
<body>

<div>
    <ul style="list-style:none;margin:0px; ">
<%--        <li style="float:left;"><a>  <img id="img"--%>
<%--                                          style="display: block;-webkit-user-select: none;margin: auto;cursor: zoom-in;background-color: hsl(0, 0%, 90%);transition: background-color 300ms;"--%>
<%--                                          src="imgSocketServlet?deviceId=123456" width="450" height="396">  </a></li>--%>
<%--        <li style="float:left; "><a>&nbsp;&nbsp;</a></li>--%>

        <li style="float:left; "><a><img id="img1"
                                          style="display: block;-webkit-user-select: none;cursor: zoom-in;background-color: hsl(0, 0%, 90%);transition: background-color 300ms;"
                                          src="getDepthImgServlet?deviceId=123456" width="1280" height="640">   </a></li>
        <%--<li style="float:left; "><a>&nbsp;&nbsp;</a></li>
        <li style="float:left; "><a> <img id="img2"
                                           style="display: block;-webkit-user-select: none;cursor: zoom-in;background-color: hsl(0, 0%, 90%);transition: background-color 300ms;"
                                           src="getDepthImgServlet?deviceId=123456" width="450" height="396">  </a></li>--%>
</ul>
</div>






<script src="${pageContext.request.contextPath}/statics/layui/lib/jquery-3.4.1/jquery-3.4.1.min.js"
        charset="utf-8"></script>
<script src="${pageContext.request.contextPath}/statics/layui/lib/layui-v2.5.5/layui.js" charset="utf-8"></script>
<script src="${pageContext.request.contextPath}/statics/layui/lib/jq-module/jquery.particleground.min.js"
        charset="utf-8"></script>


<script type="text/javascript">
    //图片自动切换部分：
    function qiehuan() {
        let random = (((1+Math.random())*new Date().getTime())|0).toString(16);
        //var imgs = "GetImgServlet?deviceId=123456&time="+random; /*图片的地址 */
        var imgColor = "imgSocketServlet?deviceId=123456&time="+random;
        var imgDepth = "getDepthImgServlet?deviceId=123456&time="+random;
        // document.getElementById("img").src = imgColor;
        document.getElementById("img1").src = imgDepth;
        //document.getElementById("img2").src = imgs;
    }
    setInterval("qiehuan()", 40);  //每25ms重新运行函数qiehuan()------帧率为1000/25=40fps

</script>


</body>
</html>
