package com.glass.util;

import java.util.Scanner;

public class Calculate {
    private final static double EARTH_RADIUS = 6378.137;

    private static double rad(double d) {
        return d * Math.PI / 180.0;
    }

    /**
     * 根据两点间经纬度坐标（double值），计算两点间距离，单位为米
     * 先输入第一个点的纬度、经度； 再输第二个的。
     * @param lat1  第一个纬度
     * @param lng1  第一个经度
     * @param lat2
     * @param lng2
     * @return
     */
    public static double GetDistance(double lat1, double lng1, double lat2, double lng2) {
        double radLat1 = rad(lat1);
        double radLat2 = rad(lat2);
        double a = radLat1 - radLat2;
        double b = rad(lng1) - rad(lng2);
        double s = 2 * Math.asin(Math.sqrt(
                Math.pow(Math.sin(a / 2), 2) + Math.cos(radLat1) * Math.cos(radLat2) * Math.pow(Math.sin(b / 2), 2)));
        s = s * EARTH_RADIUS;
        s = (s * 10000) / 10;
        return s;
    }

    public static void main(String[] args) {
        while(true){
            Scanner sc = new Scanner(System.in);
            double lat1 = sc.nextDouble();
            double lng1 = sc.nextDouble();
            double lat2 = sc.nextDouble();
            double lng2 = sc.nextDouble();
            System.out.println("距离差" + GetDistance(lat1, lng1,lat2, lng2) + "米");

        }
    }

}
