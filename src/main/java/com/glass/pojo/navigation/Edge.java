package com.glass.pojo.navigation;

import java.util.List;

public class Edge {
    private String roadName;
    private double roadLength;
    private double roadWidth;
    private int roadType;   //道路种类
    private int roadTypeWeight; //种类权值
    private int traffic;    //人流量
    private Node node1;
    private String nodeId1;
    private Node node2;
    private String nodeId2;
    private double weight;

    /**
     * mybatis 构造实体类
     */
    public Edge(int roadId,String roadName, double roadWidth,double roadLength,  int roadType , String nodeId1,String nodeId2, int traffic) {
        this.roadName = roadName;
        this.roadLength = roadLength;
        this.roadWidth = roadWidth;
        this.roadType = roadType;
        this.nodeId1 = nodeId1;
        this.nodeId2 = nodeId2;

        switch (this.roadType) {
            case 1: //  1是普通道路
                this.roadTypeWeight = 2;
                break;
            case 2: //行人道路
                this.roadTypeWeight = 1;
                break;
            case 3: //含阶梯的道路
                this.roadTypeWeight = 20;
                break;
            case 4: //其他不适合视障人士行走的道路
                this.roadTypeWeight = 15;
                break;
            default: this.roadTypeWeight = 2;
        }

        iniWeight();
    }

    public void setNode(List<Node> nodes){
        for(Node node:nodes){
            if(node.getId().equals(nodeId1)){
                setNode1(node);
            }else if (node.getId().equals(nodeId2)){
                setNode2(node);
            }
        }
    }

    Edge(Node node1,Node node2,double weight){
        this.node1 = node1;
        this.node2 = node2;
        this.weight = weight;
    }


    public Node getOtherNode(Node node){
        if(node.equals(node1) )return node2;
        else if(node.equals(node2)) return node1;
        else return null;
    }

    public void iniWeight(){
        weight = roadLength*roadTypeWeight + traffic + roadLength/roadWidth;
    }
    public Edge(Node node1,Node node2,String roadName, double roadLength, double roadWidth, int roadType) {
        this.roadName = roadName;
        this.roadLength = roadLength;
        this.roadWidth = roadWidth;
        this.roadType = roadType;
        this.node1 = node1;
        this.node2 = node2;

        switch (this.roadType) {
            case 1: //  1是普通道路
                this.roadTypeWeight = 2;
                break;
            case 2: //行人道路
                this.roadTypeWeight = 1;
                break;
            case 3: //含阶梯的道路
                this.roadTypeWeight = 20;
                break;
            case 4: //其他不适合视障人士行走的道路
                this.roadTypeWeight = 15;
                break;
            default: this.roadTypeWeight = 2;
        }

        iniWeight();
    }

    @Override
    public String toString() {
        return "Edge{" +
                "roadName='" + roadName + '\'' +
                ", roadLength=" + roadLength +
                ", roadWidth=" + roadWidth +
                ", roadType=" + roadType +
                ", roadTypeWeight=" + roadTypeWeight +
                ", traffic=" + traffic +
                '}';
    }

    public double getWeight() {
        return weight;
    }

    public Node getNode1() {
        return node1;
    }

    public void setNode1(Node node1) {
        this.node1 = node1;
    }

    public Node getNode2() {
        return node2;
    }

    public void setNode2(Node node2) {
        this.node2 = node2;
    }

    public String getRoadName() {
        return roadName;
    }

    public void setRoadName(String roadName) {
        this.roadName = roadName;
    }

    public double getRoadLength() {
        return roadLength;
    }

    public void setRoadLength(double roadLength) {
        this.roadLength = roadLength;
    }

    public double getRoadWidth() {
        return roadWidth;
    }

    public void setRoadWidth(double roadWidth) {
        this.roadWidth = roadWidth;
    }

    public int getRoadType() {
        return roadType;
    }

    public void setRoadType(int roadType) {
        this.roadType = roadType;
    }

    public int getRoadTypeWeight() {
        return roadTypeWeight;
    }

    public void setRoadTypeWeight(int roadTypeWeight) {
        this.roadTypeWeight = roadTypeWeight;
    }

    public int getTraffic() {
        return traffic;
    }

    public void setTraffic(int traffic) {
        this.traffic = traffic;
    }
}
