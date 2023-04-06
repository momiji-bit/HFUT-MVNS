package com.glass.service;

import com.alibaba.fastjson.JSONObject;
import com.glass.mapper.NodeMapper;
import com.glass.mapper.RoadMapper;
import com.glass.pojo.navigation.Edge;
import com.glass.pojo.navigation.Graph;
import com.glass.pojo.navigation.Node;
import com.glass.util.SqlSessionFactoryUtils;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;

import java.util.ArrayList;
import java.util.List;

public class NavigationService {
    SqlSessionFactory sqlSessionFactory= SqlSessionFactoryUtils.getSqlSessionFactory();
    SqlSession sqlSession= sqlSessionFactory.openSession();
    NodeMapper nodeMapper = sqlSession.getMapper(NodeMapper.class);
    RoadMapper roadMapper = sqlSession.getMapper(RoadMapper.class);

    Graph graph = getGraph();
    public List<Node> getAllNode() {
        return nodeMapper.selectAllNode();
    }

    public Graph getGraph(){
        List<Node> allNode = getAllNode();
        List<Edge> edges = roadMapper.selectAllRoad();
        for (Edge e:edges){
            e.setNode(allNode);
        }

        System.out.println(edges.size());
        System.out.println(edges.get(1));
        Graph graph = new Graph(allNode,edges);
        return graph;
    }

    public List<List<Node>> findRoadByLandmark(String beginId , String landmark){
        Node begin = nodeMapper.selectNodeById(beginId);
        List<List<Node>> roads = new ArrayList<>();
        for (Node dest : graph.findNodesByLandmark(landmark)) {
            List<Node> road = graph.findRoad(begin, dest);
            roads.add(road);
        }
        return roads;
    }

    public List<Node> findRoad(String beginId , String endId){
        Node begin = nodeMapper.selectNodeById(beginId);
        Node dest = nodeMapper.selectNodeById(endId);
        List<Node> road = graph.findRoad(begin, dest);
        return road;
    }

    public List<Edge> findEdgeByNode(List<Node> roadNodes){
        List<Edge> e = new ArrayList<>() ;
        for(int i =0;i<roadNodes.size()-1;i++){
            e.add(graph.findEdge(roadNodes.get(i),roadNodes.get(i+1)));
        }
        return e;
    }
}
