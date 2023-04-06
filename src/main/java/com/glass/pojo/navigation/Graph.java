package com.glass.pojo.navigation;

import java.util.*;

public class Graph {
    public List<Node> nodes=new ArrayList<>();
    public Map<Node,List<Edge>>  map =new HashMap<>();
    public List<Node> notKnowNodes;
    public List<Node> knowNodes;

    public Graph(List<Node> nodes,List<Edge> edges){
        this.nodes = nodes;
        for (Node node : nodes) {
            List<Edge> nodeEdges=new ArrayList<>();
            for (Edge e:edges){
                if(e.getNode1().equals(node)||e.getNode2().equals(node)){
                    nodeEdges.add(e);
                }
            }
            this.map.put(node,nodeEdges);
        }
    }

    public Graph() {

    }

    /**
     *
     * @param nodes
     * @return 返回nodes中已访问状态的结点
     */
    public List<Node> findKnowNode(List<Node> nodes){
        List<Node> knowNodes = new ArrayList<>();
        for (Node node : nodes){
            if(node.isKnow()){
                knowNodes.add(node);
            }
        }
        return knowNodes;
    }

    /**
     *
     * @param nodes
     * @return 返回 nodes 中未访问状态的结点
     */
    public List<Node> findNotKnowNode(List<Node> nodes){
        List<Node> notKnowNodes = new ArrayList<>();
        for (Node node : nodes){
            if(!node.isKnow()){
                notKnowNodes.add(node);
            }
        }
        return notKnowNodes;
    }

    /**
     *
     * @param nodes
     * @return 返回结点列表nodes 中 到起始结点最近的一个结点
     */
    public Node findMinDistance(List<Node> nodes){
        double weight = Double.MAX_VALUE;
        Node minNode=null;
        for (Node node: nodes) {
            double tempDis = node.getDistance();
            if(tempDis<weight){
                weight = tempDis;
                minNode = node;
            }
        }
        return minNode;
    }

    /**
     *
     * @param node
     * @return node的所有相邻结点列表
     */
    public List<Node> findNearNodes(Node node){
        List<Node> nodes = new ArrayList<>();
        List<Edge> edges = map.get(node);
        for (Edge edge:edges){
            Node otherNode = edge.getOtherNode(node);
            if(otherNode!=null){
                nodes.add(otherNode);
            }
        }
        System.out.println("\tfindNearNodes:\t"+node+"\n------->\t"+nodes);
        return nodes;
    }

    public void  initial(){
        for (Node node : this.nodes) {
            node.iniNode();
        }
    }

    /**
     * node1 node2之间的边
     * @param node1
     * @param node2
     * @return
     */
    public Edge findEdge(Node node1,Node node2){
        Edge edge=null;
        for(Edge e:this.map.get(node1)){
            if(e.getOtherNode(node1).equals(node2)){
                edge = e;
            }
        }
        return edge;
    }

    /**
     *
     * @param begin 起始结点
     * @param dest 目的地结点
     * @return  中间经过的所有结点（包括起点和终点）
     */
    public List<Node> findRoad(Node begin,Node dest){
        boolean findBegin=false;
        boolean findDest=false;
        for(Node n:nodes){
            if(n.equals(begin)){
                begin = n;
                findBegin=true;
            }
            else if(n.equals(dest)){
                dest = n;
                findDest=true;
            }
        }
        if(!(findBegin&&findDest))  {
            System.out.println("not found a road ,没有找到节点");
            return new ArrayList<>();
        }
        this.initial();
        if(!nodes.contains(dest)|| map.get(dest)==null || map.get(dest).isEmpty()||!nodes.contains(begin)|| map.get(begin)==null || map.get(begin).isEmpty()){
            System.out.println("not found a road");
            return new ArrayList<>();
        }

        //初始化与起始结点连接的边
        begin.setDistance(0);
        begin.setKnow(true);
        List<Node> nearNodes = findNearNodes(begin);
        for (Node nearNode:nearNodes){
            nearNode.setDistance(findEdge(begin,nearNode).getWeight());
            nearNode.setParent(begin);
        }
        notKnowNodes = findNotKnowNode(nodes);
        knowNodes = findKnowNode(nodes);

        while(!notKnowNodes.isEmpty()){
            //从未访问的结点中取出 最短路径的点：并设为已访问

            Node minDistanceNode = findMinDistance(notKnowNodes);
            if(minDistanceNode == null){
                if(dest.getParent()==null){
                    System.out.println("minDistanceNode  id  null  Error：不存在从 "+begin.getId()+" 到 "+ dest.getId()+"的路 ；存在孤立图，且终点和起点不连通");
                    return new ArrayList<>();
                }
                System.out.println("minDistanceNode  id  null  Error：存在孤立图，终点起点连通");

                break;
            }
            minDistanceNode.setKnow(true);


            //通过最短的点刷新距离：
            double minDis = minDistanceNode.getDistance();
            nearNodes = findNearNodes(minDistanceNode);
            for (Node nearNode:nearNodes){
                double tempDis = nearNode.getDistance();
                Edge nearEdge = findEdge(minDistanceNode,nearNode);
                double newDis = minDis+nearEdge.getWeight();
                if(tempDis>newDis){
                    nearNode.setDistance(newDis);
                    nearNode.setParent(minDistanceNode);
                }
            }

            notKnowNodes = findNotKnowNode(nodes);
            knowNodes = findKnowNode(nodes);
        }

        String path = "[" + dest.getId() + "]";

        double shortWeight = dest.getDistance();
        System.out.println("shortWeight: "+shortWeight);

        List<Node> roadNodes = new ArrayList<>();
        roadNodes.add(dest);
        while ((dest.getParent()!=null)&&(!dest.equals(begin))){
            roadNodes.add(dest.getParent());
            path = "[" + dest.getParent().getId() +"] --> "+path;
            dest = dest.getParent();
        }
        System.out.println(path);
        Collections.reverse(roadNodes);

        return roadNodes;


    }

    public List<Node> findNodesByLandmark(String landmark){
        List<Node> findNodes = new ArrayList<>();
        for(Node node:this.nodes){
            if(node.getLandmark().contains(landmark) ||node.getId().equals(landmark)){
                findNodes.add(node);
            }
        }
        return findNodes;
    }


    public static void main(String[] args) {
        Graph g = new Graph();
        Node v1 = new Node("v1");
        Node v2 = new Node("v2");
        Node v3 = new Node("v3");
        Node v4 = new Node("v4");
        Node v5 = new Node("v5");
        Node v6 = new Node("v6");
        Node v7 = new Node("v7");

        List<Edge> v1List = new ArrayList<>();
        v1List.add(new Edge(v1,v2,2));
//        v1List.add(new Edge(v1,v3,4));
        v1List.add(new Edge(v1,v4,1));

        List<Edge> v2List = new ArrayList<>();
//        v2List.add(new Edge(v2,v1,2));
        v2List.add(new Edge(v2,v4,3));
        v2List.add(new Edge(v2,v5,10));

        List<Edge> v3List = new ArrayList<>();
        v3List.add(new Edge(v3,v1,4));
//        v3List.add(new Edge(v3,v4,2));
        v3List.add(new Edge(v3,v6,5));

        List<Edge> v4List = new ArrayList<>();
        v4List.add(new Edge(v4,v3,2));
        v4List.add(new Edge(v4,v6,8));
        v4List.add(new Edge(v4,v7,4));
        v4List.add(new Edge(v4,v5,2));

        List<Edge> v5List = new ArrayList<>();
        v5List.add(new Edge(v5,v7,6));

        List<Edge> v6List = new ArrayList<>();

        List<Edge> v7List = new ArrayList<>();
        v7List.add(new Edge(v7,v6,1));


        g.nodes.add(v1);
        g.nodes.add(v2);
        g.nodes.add(v3);
        g.nodes.add(v4);
        g.nodes.add(v5);
        g.nodes.add(v6);
        g.nodes.add(v7);


        g.map.put(v1,v1List);
        g.map.put(v2,v2List);
        g.map.put(v3,v3List);
        g.map.put(v4,v4List);
        g.map.put(v5,v5List);
        g.map.put(v6,v6List);
        g.map.put(v7,v7List);


        List<Node> road = g.findRoad(v1, v6);

    }



}
