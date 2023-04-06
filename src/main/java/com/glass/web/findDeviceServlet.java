package com.glass.web;

import com.alibaba.fastjson.JSONObject;
import com.glass.pojo.Device;
import com.glass.service.DeviceService;

import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;
import java.io.IOException;
import java.io.PrintWriter;

@WebServlet("/findDeviceByDeviceId.do")
public class findDeviceServlet extends HttpServlet {
    DeviceService deviceService = new DeviceService();
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String deviceId = request.getParameter("deviceId");
        Device device = deviceService.getDeviceByDeviceid(deviceId);
        System.out.println(device);
        JSONObject json = new JSONObject();
        json.put("collide",device.getCollide().equals("1"));
        json.put("edge",device.getEdge().equals("1"));
        json.put("gesture","1".equals(device.getGesture()));
        json.put("broadcast","1".equals(device.getBroadcast()));
        json.put("navigation","1".equals(device.getNavigation()));
        response.setContentType("text/json;charset=UTF-8");
        response.setCharacterEncoding("UTF-8");
        PrintWriter out = response.getWriter();
        out.println(json.toJSONString());
        out.flush();
        out.close();
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doGet(request, response);
    }
}
