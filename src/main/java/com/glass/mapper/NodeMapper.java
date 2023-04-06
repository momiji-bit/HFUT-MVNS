package com.glass.mapper;

import com.glass.pojo.navigation.Node;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface NodeMapper {
    @Select("SELECT * FROM node")
    public List< Node>  selectAllNode();

    @Select("SELECT * FROM node where node_id = #{id}")
    public Node selectNodeById(String id);
}
