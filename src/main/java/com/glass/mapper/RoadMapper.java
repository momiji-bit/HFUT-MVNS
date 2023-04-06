package com.glass.mapper;

import com.glass.pojo.navigation.Edge;
import com.glass.pojo.navigation.Node;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface RoadMapper {
    @Select("SELECT * FROM road")
    public List<Edge> selectAllRoad();

    @Select("SELECT * FROM road WHERE node_id1=#{node.getId()} or node_id2=#{node.getId()}")
    public List<Edge> selectAllRoadByNode(Node node);
}
