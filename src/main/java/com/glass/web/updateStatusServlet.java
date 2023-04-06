package com.glass.web;

import com.alibaba.fastjson.JSONObject;
import com.glass.service.DeviceService;

import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;
import java.io.*;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Arrays;
import java.util.Map;

/**
 *
 */
@WebServlet("/updateStatus.do")
public class updateStatusServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)  {
        System.out.println("当前正在更改设备状态");
        String device = request.getParameter("device");
        String collide = request.getParameter("collide");
        String edge = request.getParameter("edge");
        String gesture = request.getParameter("gesture");
        String broadcast = request.getParameter("broadcast");
        String navigation = request.getParameter("navigation");

        Map<String, String[]> parameterMap = request.getParameterMap();
        System.out.println("************map************");
        JSONObject jsonParm = new JSONObject();
        for (String s:parameterMap.keySet()
             ) {
            jsonParm.put(s,parameterMap.get(s)[0]);
        }
        System.out.println(jsonParm.toJSONString());
        //使用socket将数据发送到服务器
        boolean res;
        try (Socket socket = new Socket("127.0.0.1", 8888)) {
            OutputStream os = socket.getOutputStream();
            PrintWriter pw = new PrintWriter(os);
            pw.print(jsonParm);
            pw.flush();

            socket.shutdownOutput();
            os.close();
            pw.close();
            res = new DeviceService().updateStatusByDeviceId(device, collide, edge, gesture, broadcast, navigation);

        } catch (UnknownHostException e) {
            res=false;
            e.printStackTrace();
        } catch (IOException e) {
            res=false;
            e.printStackTrace();
        }

        JSONObject json = new JSONObject();
        json.put("success",res);
        System.out.println(json.toJSONString());
        response.setContentType("text/json;charset=UTF-8");
        response.setCharacterEncoding("UTF-8");
        try {
            PrintWriter out = response.getWriter();
            out.println(json);
            out.flush();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }



    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doGet(request, response);
    }
}
