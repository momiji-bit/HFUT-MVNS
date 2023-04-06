package com.glass.pojo;

public class Device {
    String id;
    String device_id;
    String collide;
    String edge;
    String gesture;
    String broadcast;
    String navigation;
    String flag2;
    String flag3;
    String flag4;
    String flag5;
    String flag6;
    String flag7;

    public Device(String id, String device_id, String collide, String edge, String gesture, String broadcast, String navigation, String flag2, String flag3, String flag4, String flag5, String flag6, String flag7) {
        this.id = id;
        this.device_id = device_id;
        this.collide = collide;
        this.edge = edge;
        this.gesture = gesture;
        this.broadcast = broadcast;
        this.navigation = navigation;
        this.flag2 = flag2;
        this.flag3 = flag3;
        this.flag4 = flag4;
        this.flag5 = flag5;
        this.flag6 = flag6;
        this.flag7 = flag7;
    }

    @Override
    public String toString() {
        return "Device{" +
                "id='" + id + '\'' +
                ", device_id='" + device_id + '\'' +
                ", collide='" + collide + '\'' +
                ", edge='" + edge + '\'' +
                ", gesture='" + gesture + '\'' +
                ", broadcast='" + broadcast + '\'' +
                ", navigation='" + navigation + '\'' +
                ", flag2='" + flag2 + '\'' +
                ", flag3='" + flag3 + '\'' +
                ", flag4='" + flag4 + '\'' +
                ", flag5='" + flag5 + '\'' +
                ", flag6='" + flag6 + '\'' +
                ", flag7='" + flag7 + '\'' +
                '}';
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getDevice_id() {
        return device_id;
    }

    public void setDevice_id(String device_id) {
        this.device_id = device_id;
    }

    public String getCollide() {
        return collide;
    }

    public void setCollide(String collide) {
        this.collide = collide;
    }

    public String getEdge() {
        return edge;
    }

    public void setEdge(String edge) {
        this.edge = edge;
    }

    public String getGesture() {
        return gesture;
    }

    public void setGesture(String gesture) {
        this.gesture = gesture;
    }

    public String getBroadcast() {
        return broadcast;
    }

    public void setBroadcast(String broadcast) {
        this.broadcast = broadcast;
    }

    public String getNavigation() {
        return navigation;
    }

    public void setNavigation(String navigation) {
        this.navigation = navigation;
    }

    public String getFlag2() {
        return flag2;
    }

    public void setFlag2(String flag2) {
        this.flag2 = flag2;
    }

    public String getFlag3() {
        return flag3;
    }

    public void setFlag3(String flag3) {
        this.flag3 = flag3;
    }

    public String getFlag4() {
        return flag4;
    }

    public void setFlag4(String flag4) {
        this.flag4 = flag4;
    }

    public String getFlag5() {
        return flag5;
    }

    public void setFlag5(String flag5) {
        this.flag5 = flag5;
    }

    public String getFlag6() {
        return flag6;
    }

    public void setFlag6(String flag6) {
        this.flag6 = flag6;
    }

    public String getFlag7() {
        return flag7;
    }

    public void setFlag7(String flag7) {
        this.flag7 = flag7;
    }
}
