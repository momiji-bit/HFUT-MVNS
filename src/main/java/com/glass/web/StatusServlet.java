package com.glass.web;

import com.glass.service.DeviceService;

import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;
import java.io.IOException;
import java.util.List;

/**
 * 获取所有deviceId
 */
@WebServlet("/status.do")
public class StatusServlet extends HttpServlet {
    DeviceService deviceService = new DeviceService();
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        List<String> allDeviceId = deviceService.getAllDeviceId();
        request.setAttribute("deviceIds",allDeviceId);
        request.getRequestDispatcher("pages/admin/status.jsp").forward(request,response);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doGet(request, response);
    }
}
