package com.glass.web;

import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.serializer.SerializerFeature;
import com.glass.pojo.navigation.Edge;
import com.glass.pojo.navigation.Node;
import com.glass.service.NavigationService;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * 根据结点id查询从begin,根据landmark查询end， 返回从begin到end的所有最小权值道路。
 * 返回的是json
 * {
 *     nodes[][]:=[node1[],node2[]]
 *     roads[][] = [edge1[],edge2[]]
 * }
 */
@WebServlet("/findRoadByLandmark")
public class FindRoadByLandmarkServlet extends HttpServlet {
    NavigationService navigationService = new NavigationService();
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String beginId = request.getParameter("beginId");
        String landmark = request.getParameter("landmark");
        List<List<Node>> roadsNodes = navigationService.findRoadByLandmark(beginId, landmark);
        List<List<Edge>> roadsEdges=new ArrayList<>();
        for (List<Node> road : roadsNodes) {
            List<Edge> edges = navigationService.findEdgeByNode(road);
            roadsEdges.add(edges);
        }

        JSONObject json = new JSONObject();
        json.put("nodes", roadsNodes);
        json.put("roads", roadsEdges);

        response.setContentType("text/json;charset=UTF-8");
        response.setCharacterEncoding("UTF-8");
        PrintWriter out = response.getWriter();
        out.println(json.toJSONString(json,SerializerFeature.DisableCircularReferenceDetect));

        System.out.println(json.toJSONString(json,SerializerFeature.DisableCircularReferenceDetect));
        out.flush();
        out.close();

    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doGet(request, response);
    }
}
