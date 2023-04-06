package com.glass.pojo.navigation;

import java.util.List;
import java.util.Objects;

public class Node {
    private String id;    //节点编号
    private double longitude;   //经度
    private double latitude;    //纬度
    private List<String> landmark;  //附近建筑
    private Node parent;
    private double distance;
    private boolean know;

    Node(String id){
        iniNode();
        this.id = id;
    }
    public Node(String id, double longitude, double latitude, String landmark) {
        this.id = id;
        this.longitude = longitude;
        this.latitude = latitude;
        this.landmark = List.of(landmark.split("、"));
        iniNode();
    }
    public Node(String id, double longitude, double latitude, List<String> landmark) {
        this.id = id;
        this.longitude = longitude;
        this.latitude = latitude;
        this.landmark = landmark;
        iniNode();
    }

    void iniNode(){
        parent = null;
        know = false;
        distance = Double.MAX_VALUE;
    }

    public boolean hasLandmark(String name){
        for (String  buildingName:this.landmark) {
            if(name.equals(buildingName))   return true;
        }
        return false;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Node node = (Node) o;
        return id.equals(node.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Node{" +
                "id='" + id + '\'' +
                ", longitude=" + longitude +
                ", latitude=" + latitude +
                ", landmark=" + landmark +
                '}';
    }

    public boolean isKnow() {
        return know;
    }

    public void setKnow(boolean know) {
        this.know = know;
    }

    public double getDistance() {
        return distance;
    }

    public void setDistance(double distance) {
        this.distance = distance;
    }

    public Node getParent() {
        return parent;
    }

    public void setParent(Node parent) {
        this.parent = parent;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public double getLongitude() {
        return longitude;
    }

    public void setLongitude(double longitude) {
        this.longitude = longitude;
    }

    public double getLatitude() {
        return latitude;
    }

    public void setLatitude(double latitude) {
        this.latitude = latitude;
    }

    public List<String> getLandmark() {
        return landmark;
    }

    public void setLandmark(List<String> landmark) {
        this.landmark = landmark;
    }

}
