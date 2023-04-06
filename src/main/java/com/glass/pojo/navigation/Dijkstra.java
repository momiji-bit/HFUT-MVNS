package com.glass.pojo.navigation;
import java.util.*;
public class Dijkstra {

    //定义顶点Vertex类

    static class Vertex{
        private final static int infinite_dis = Integer.MAX_VALUE;

        private String name;   //节点名字
        private boolean known;  //此节点是否已知

        private int adjuDist;   //此节点距离

        private Vertex parent;   //当前从初始化节点到此节点的最短路径下的父亲节点

        public Vertex(){
            this.known = false;
            this.adjuDist = infinite_dis;
            this.parent = null;
        }

        public Vertex(String name){
            this();
            this.name = name;
        }

        public String getName(){
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public boolean isKnown(){
            return known;
        }

        public void setKnown(boolean known){
            this.known = known;
        }

        public int getAdjuDist(){
            return adjuDist;
        }

        public void setAdjuDist(int adjuDist){
            this.adjuDist = adjuDist;
        }


        public Vertex getParent(){
            return parent;
        }

        public void setParent(Vertex parent){
            this.parent = parent;
        }


        public boolean equals(Object obj){
            if(!(obj instanceof Vertex)){
                throw new ClassCastException("an object to compare with a Vertex");
            }

            if(this.name==null){
                throw new NullPointerException("name of Vertex to be compared cannot be null!");
            }
            return this.name.equals(((Vertex)obj).getName());
        }


    }


    //定义有向边类
    static class Edge{
        //此有向边的起始点
        private Vertex startVertex;
        //此有向边的终点
        private Vertex endVertex;
        //此有向边的权值
        private int weight;


        public Edge(Vertex startVertex,Vertex endVertex,int weight){
            this.startVertex = startVertex;
            this.endVertex = endVertex;
            this.weight = weight;
        }

        public Vertex getStartVertex(){
            return startVertex;
        }


        public Vertex getEndVertex(){
            return endVertex;
        }

        public int getWeight(){
            return weight;
        }


        public void setStartVertex(Vertex startVertex){
            this.startVertex = startVertex;
        }

        public void setEndVertex(Vertex endVertex){
            this.endVertex = endVertex;
        }

        public void setWeight(int weight){
            this.weight = weight;
        }


    }



    private List<Vertex> vertexList; //图的顶点集

    private Map<Vertex,List<Edge> > ver_edgeList_map;  //图的每个顶点对应的有向边


    public Dijkstra(List<Vertex> vertexList, Map<Vertex, List<Edge>> ver_edgeList_map) {
        this.vertexList = vertexList;
        this.ver_edgeList_map = ver_edgeList_map;
    }


    public void setRoot(Vertex v){
        v.setParent(null);
        v.setAdjuDist(0);
    }


    //从初始节点开始递归更新邻接表

    private void updateChildren(Vertex v){

        if (v==null){
            return;
        }
        if (ver_edgeList_map.get(v)==null || ver_edgeList_map.get(v).size()==0){
            return;
        }


        List<Vertex> childrenList = new LinkedList<Vertex>();

        for (Edge e:ver_edgeList_map.get(v)){

            Vertex childVertex = e.getEndVertex();

            //如果子节点之前未知,则把当前子节点加入更新列表

            if (!childVertex.isKnown()){
                childVertex.setKnown(true);
                childVertex.setAdjuDist(v.getAdjuDist()+e.getWeight());
                childVertex.setParent(v);
                childrenList.add(childVertex);
            }

            //子节点之前已知,则比较子节点的adjudist&&nowDist

            int nowDist = v.getAdjuDist() + e.getWeight();
            if (nowDist >= childVertex.getAdjuDist()){
                continue;
            }else {
                childVertex.setAdjuDist(nowDist);
                childVertex.setParent(v);
            }
        }
        //更新每一个子节点

        for (Vertex vc:childrenList){
            updateChildren(vc);
        }



    }

    /**
     *
     * @param startIndex   dijkstra遍历的起点节点下标
     * @param destIndex    dijkstra遍历的终点节点下标
     */
    public void dijkstraTravasal(int startIndex,int destIndex){

        Vertex start = vertexList.get(startIndex);
        Vertex dest  = vertexList.get(destIndex);
        String path = "[" + dest.getName() + "]";

        setRoot(start);

        updateChildren(vertexList.get(startIndex));

        int shortest_length = dest.getAdjuDist();

        while ((dest.getParent()!=null)&&(!dest.equals(start))){
            path = "[" + dest.getParent().getName() +"] --> "+path;
            dest = dest.getParent();
        }

        System.out.println("["+vertexList.get(startIndex).getName()+"] to ["+
                vertexList.get(destIndex).getName()+"] dijkstra shortest path:: "+path);

        System.out.println("shortest length::" + shortest_length);
    }






    public static void main(String[] args) {

        Vertex a= new Vertex("a");
        Vertex b= new Vertex("b");
        Vertex c= new Vertex("c");
        Vertex d= new Vertex("d");
        Vertex e= new Vertex("e");
        Vertex f= new Vertex("f");
        Vertex g= new Vertex("g");


        List<Vertex> verList = new LinkedList<Dijkstra.Vertex>();
        verList.add(a);
        verList.add(b);
        verList.add(c);
        verList.add(d);
        verList.add(e);
        verList.add(f);
        verList.add(g);



        Map<Vertex, List<Edge>> vertex_edgeList_map = new HashMap<Vertex, List<Edge>>();

        List<Edge> v1List = new LinkedList<Dijkstra.Edge>();
        v1List.add(new Edge(a,b,2));
        v1List.add(new Edge(a,d,1));


        List<Edge> v2List = new LinkedList<Dijkstra.Edge>();
        v2List.add(new Edge(b,d,3));
        v2List.add(new Edge(b,e,10));


        List<Edge> v3List = new LinkedList<Dijkstra.Edge>();
        v3List.add(new Edge(c,a,4));
        v3List.add(new Edge(c,f,5));

        List<Edge> v4List = new LinkedList<Dijkstra.Edge>();
        v4List.add(new Edge(d,c,2));
        v4List.add(new Edge(d,e,2));
        v4List.add(new Edge(d,f,8));
        v4List.add(new Edge(d,g,4));

        List<Edge> v5List = new LinkedList<Dijkstra.Edge>();
        v5List.add(new Edge(e,g,6));

        List<Edge> v6List = new LinkedList<Dijkstra.Edge>();

        List<Edge> v7List = new LinkedList<Dijkstra.Edge>();
        v7List.add(new Edge(g,f,1));

        vertex_edgeList_map.put(a, v1List);
        vertex_edgeList_map.put(b, v2List);
        vertex_edgeList_map.put(c, v3List);
        vertex_edgeList_map.put(d, v4List);
        vertex_edgeList_map.put(e, v5List);
        vertex_edgeList_map.put(f, v6List);
        vertex_edgeList_map.put(g, v7List);


        Dijkstra gra = new Dijkstra(verList, vertex_edgeList_map);
//      g.dijkstraTravasal(1, 5);
        gra.dijkstraTravasal(0, 6);




    }





}




