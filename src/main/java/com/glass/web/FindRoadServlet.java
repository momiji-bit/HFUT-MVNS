package com.glass.web;

import com.alibaba.fastjson.JSONObject;
import com.glass.pojo.navigation.Edge;
import com.glass.pojo.navigation.Graph;
import com.glass.pojo.navigation.Node;
import com.glass.service.NavigationService;

import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/**
 * 根据结点id查询从begin到end 的最小权值道路。
 * 返回的是json
 * {
 *     nodes：{}
 *     roads：{}
 * }
 */
@WebServlet("/findRoad")
public class FindRoadServlet extends HttpServlet {
    NavigationService navigationService = new NavigationService();
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String begin = request.getParameter("begin");
        String end = request.getParameter("end");
        List<Node> nodes = navigationService.findRoad(begin, end);
        List<Edge> edges = navigationService.findEdgeByNode(nodes);
        JSONObject json = new JSONObject();
        json.put("nodes", nodes);
        json.put("roads", edges);

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
