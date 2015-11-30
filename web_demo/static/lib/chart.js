var margin = {top: 40, right: 30, bottom: 100, left: 60},
    width = screen.width/2.1 - margin.left - margin.right,
    height = screen.height/2.1 - margin.top - margin.bottom;

d3_logFormat = function(d) {
                var x = Math.log(d) / Math.log(10) + 1e-6;
                    return Math.abs(x - Math.floor(x)) < .48 ? d3.format(",.2s")(d) : "";
            }   

function getTimeScale(startDate, stopDate) {
	return d3.time.scale().range([0, width]).domain([startDate, stopDate]);
}


function getLinearScale(range, domain) {
	return d3.scale.linear().range(range).domain(domain).nice();
}

function getLogScale(range, domain) {
	return d3.scale.log().range(range).domain(domain).nice();
}

function getOrdinalScale(range) {
	return d3.scale.ordinal().domain(d3.range(range)).rangeBands([0, width], .1)
}

function getMinsMaxs(array, fields, inout) {
	array.forEach(function(v){
		fields.forEach(function(f) {
			if (inout['min_'+f] == undefined || inout['min_'+f] > v[f]) {
				inout['min_'+f] = v[f];
			}
			if (inout['max_'+f] == undefined || inout['max_'+f] < v[f]) {
				inout['max_'+f] = v[f];
			}
		});
	});
}

function get_random_color(name) {
	if (name == "M") return "blue"; //"#90EE90";
	if (name == "K") return "pink";
	if (name == "parallel") return "#A020F0";
	if (name == "urgent") return "#FF2A00";
	if (name == "qtime" || name == "exit") return "#FF2A00";
	if (name == "runtime") return "#90EE90";
	if (name == "tbtl" ) return "#FFA500";
	if (name == "< 10 sec") return "#DFFFA5";
	if (name == "< 1 min") return "#B3EE3A";
	if (name == "< 5 min") return "#FFFF66";
	if (name == "< 15 min") return "#FFCC00";
	if (name == "< 1 hr") return "#FF9900";
	if (name == "> 1 hr") return "#E3170D";
		
	Math.seedrandom(name);
    var letters = '0123456789ABCDEF'.split('');
    var color = '#';
    for (var i = 0; i < 6; i++ ) {
        color += letters[Math.round(Math.random() * 15)];
    }
    return color;
}


function BaseChart(options) {
	for (var o in options) { 
		this[o] = options[o]; 
	}
        this.draw();
        
}

BaseChart.prototype.draw = function() {
    	this.createSVG();
	this.appendTitle();
	this.appendGridding();
	this.appendGroups();
	this.appendLegend();
	this.events();
	this.appendXAxis();
        this.appendYAxis();
		
        for (var s in this.shapes) {
		this["draw_"+s]();
	}
}

BaseChart.prototype.createSVG = function() {
	this.svg = d3.select(this.div).append("svg")
	.attr("width", width + margin.left + margin.right)
	.attr("height", height + margin.top + margin.bottom)
	.append("g")
	.attr("transform", "translate(" + margin.left + "," + margin.top + ")");
};

BaseChart.prototype.appendTitle = function() {
	this.svg.append("text")
		.attr("x", (width / 2))       
        .attr("y", 0 - (margin.top / 3))
        .attr("text-anchor", "middle")  
        .style("font-size", "16px") 
        .style("text-decoration", "underline")  
        .text(this.title);
};

BaseChart.prototype.appendGridding = function() {
    var xAxis = d3.svg.axis().scale(this.xScale).orient("bottom").tickSize(-height, 0, 0);
    this.svg.append("g")         
        .attr("class", "grid")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis.tickFormat(""));
    var yAxis = d3.svg.axis().scale(this.yScale).orient("left").tickSize(-width, 0, 0);
    this.svg.append("g")         
        .attr("class", "grid")
        .call(yAxis.tickFormat(""));
}

BaseChart.prototype.appendGroups = function() {
	for (var s in this.shapes) {
            	this[s] = this.svg.selectAll("."+s)
			.data(this.shapes[s])
			.enter().append("g")
			.attr("class", "g");
	}
}

BaseChart.prototype.events = function() {
	if (this.url) {
		for (var s in this.shapes) {
			var url = this.url;
			this[s].on("mouseout", function(){d3.select(this).style("opacity", 1);})
			this[s].on("mouseover", function(){d3.select(this)
				.style("opacity", 0.5);})
				.style("cursor", "pointer"); 
			this[s].on("click", function(d){
				window.location.href = "http://" + window.location.host + "/" + url + "/"+ d.name;
			});
		}
	}
}

BaseChart.prototype.appendXAxis = function() {
    var xAxis = d3.svg.axis().scale(this.xScale).orient("bottom").tickFormat(this.xFormat);
    if (this.rotate_x_ticks) {
		this.svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .text(this.xLabel)
        .call(xAxis)
        .selectAll("text")
			.style("text-anchor", "end")
            .attr("dx", "-.7em")
            .attr("dy", ".15em")
            .attr("transform", function(d) {
				return "rotate(-35)" 
            })
	} else {
    this.svg.append("g")
		.attr("class", "x axis")
		.attr("transform", "translate(0," + height + ")")
		.call(xAxis)
		.append("text")
			.style("text-anchor", "end")
			.attr("x", width - 10)
			.attr("y", -6)
			.attr("dx", ".71em")
			.attr("dy", ".15em")
		.text(this.xLabel);
	}
}

BaseChart.prototype.appendYAxis = function() {
    var yAxis = d3.svg.axis().scale(this.yScale).orient("left").tickFormat(this.yFormat);
    this.svg.append("g")
		.attr("class", "y axis")
		.attr("transform", "translate(0, 0)")
		.call(yAxis)
		.append("text")
		.attr("class", "label")
		.attr("transform", "rotate(-90)")
		.attr("y", 6)
		.attr("dy", ".71em")
		.style("text-anchor", "end")
		.text(this.yLabel);
}

BaseChart.prototype.appendLegend = function() {
	if (this.legend) {
		for (s in this.shapes) {
			var legend = this.svg.selectAll("legend")
				.data(this.shapes[s].reverse())
				.enter().append("g")
				.attr("class", "legend")
				.attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });
		
			legend.append("rect")
				.attr("x", margin.left + 35)
				.attr("width", 18)
				.attr("height", 18)
				.style("fill", function(d) { return get_random_color(d.name);});

			legend.append("text")
				.attr("x", margin.left + 30)
				.attr("y", 9)
				.attr("dy", ".35em")
				.style("text-anchor", "end")
				.text(function(d) { return d.name; });
		}
	}
}


BaseChart.prototype.draw_lines = function(color) {
    	var xScale = this.xScale;
	var yScale = this.yScale;
	
	var line = d3.svg.line()
	.defined(function(d) {return isFinite(yScale(d.y));})
    .x(function(d) { return xScale(d.x); })
    .y(function(d) { return yScale(d.y); }); 
	
	this.lines.append("path")
		.attr("class", "line")
		.attr("d", function(d) {return line(d.data); })
		.style("stroke", function(d) { return (color)?color:get_random_color(d.name);})
		.style("fill", "none")
		.style("stroke-width", function(d) {return d.stroke_width == undefined ? 4 : d.stroke_width;})
		.style("stroke-dasharray", function(d) {if (d.future == true) return ("3, 3"); });
		
	this.lines.append("title")
		.text(function(d) {return d.title;});
}

BaseChart.prototype.draw_rectangles_orig = function() {
	var xScale = this.xScale;
	var yScale = this.yScale;
	var h = (height/yScale.domain()[1]);
	var total_secs = (xScale.domain()[1]-xScale.domain()[0])/1000;
        console.log(xScale.domain());
        console.log(yScale.domain());
	this.rectangles.append("rect")
			.attr("x", function(d) { return xScale(d.x);})
			.attr("y", function(d) { return yScale(0)-(d.h*h)})
			.attr("height", function(d) { return d.h*h;})
			.attr("width", function(d) { return d.w*width/total_secs;})
			.style("fill", function(d) { return get_random_color(d.name); });

    this.rectangles.append("title")
		.text(function(d) { return d.title;});
}

BaseChart.prototype.draw_rectangles = function() {
	var xScale = this.xScale;
	var yScale = this.yScale;
	var h = (height/yScale.domain()[1]);
	console.log(xScale.domain());
        console.log(yScale.domain());
	this.rectangles.append("rect")
			.attr("x", function(d) { return xScale(d.x);})
			.attr("y", function(d) { return yScale(d.h)})
			.attr("height", function(d) { return d.h*h;})
			.attr("width", function(d) { return d.w;})
			.style("fill", function(d) { return get_random_color(d.name); });

    this.rectangles.append("title")
		.text(function(d) { return d.title;});
}


BaseChart.prototype.draw_areas = function() {
	var xScale = this.xScale;
	var yScale = this.yScale;
	var area = d3.svg.area()
		.defined(function(d) {return yScale(d.y0 -d.y1) > 0;})
		.x(function(d) { return xScale(d.x); })
		.y0(function(d) { return yScale(d.y0); })
		.y1(function(d) { return yScale(d.y1); });

	this.areas.append("path")
			.attr("class", "area")
			.attr("d", function(d) { return area(d.data); })
			.style("fill", function(d) { return get_random_color(d.name); });
	this.areas.append("title")
		.text(function(d) {return d.name;});
}

BaseChart.prototype.draw_stackedAreas = function() {
	var xScale = this.xScale;
	var yScale = this.yScale;
	var total = 0;
	this.shapes.stackedAreas.forEach(function (d) { d.total = d3.sum( d.data.map(function (f) {return f.y;})); total += d.total;});
	
	var stacked_area = d3.svg.area()
		.x(function(d) { return xScale(d.x); })
		.y0(function(d) { return yScale(d.y0); })
		.y1(function(d) { return yScale(d.y0+d.y); });

	this.stackedAreas.append("path")
      .attr("class", "area")
      .attr("d", function(d) {return stacked_area(d.data); })
      .style("fill", function(d) { return get_random_color(d.name); });
      
    this.stackedAreas.append("title")
		.text(function(d) {return d.title + " (" + d3.format(".1%")(d.total/total) + ")";});
}

BaseChart.prototype.draw_bars = function() {
	var xScale = this.xScale;
	var yScale = this.yScale;
	var total=0;
	this.shapes.bars.forEach(function(d) { total += d.y; });
	
	this.bars.append("rect")
      .attr("x", function(d) {return xScale(d.name);})
      .attr("width", xScale.rangeBand())
      .attr("y", function(d) {return yScale(d.y); })
      .attr("height", function(d) {return height - yScale(d.y);})
      .style("fill", function(d) { return get_random_color(d.name); });

	this.bars.append("title")
      .text(function(d) { 
			  var s = d.name;
              s+="\n" + d3.format(".1%")(d.y/total);
              s+= "\n" + d3.format(".2s")(d.y);
              return s;});
}

BaseChart.prototype.draw_gradientLines = function(stroke_width) {
	var xScale = this.xScale;
	var yScale = this.yScale;
	var line = d3.svg.line()
	//    .interpolate("bundle")
	//    .interpolate("cardinal-open")
	//    .interpolate("monotone")
	    .defined(function(d) {return (d.y != undefined);})
//	    .defined(function(d) {return d.y > 0;})
	    .x(function(d) { return xScale(d.x); })
	    .y(function(d) { return yScale(d.y); }); 
	    
	var color = d3.scale.ordinal().range(["#E3170D","#FFA000","#ffF800","chartreuse","#736AFF"]);
    color.domain(["poor", "mediocre","ok","good", "speedup"]);                     
                          
	this.gradientLines.append("linearGradient")
      .attr("id", "line-gradient")
      .attr("gradientUnits", "userSpaceOnUse")
      .attr("x1", 0).attr("y1", yScale(0))
      .attr("x2", 0).attr("y2", yScale(4))
    .selectAll("stop")
      .data([
        {offset: "0%", color: "#E3170D"},
        {offset: "6.25%", color: "#E3170D"},
        {offset: "6.25%", color: "#FFA000"},
        {offset: "12.5%", color: "#FFA000"},
        {offset: "12.5%", color: "#ffF800"},
        {offset: "18.75%", color: "#ffF800"},
        {offset: "18.75%", color: "chartreuse"},
        {offset: "25%", color: "chartreuse"},
        {offset: "25%", color: "#736AFF"},
        {offset: "100%", color: "#736AFF"}
      ])
    .enter().append("stop")
      .attr("offset", function(d) { return d.offset; })
      .attr("stop-color", function(d) { return d.color; });

	this.gradientLines.append("path")
      .attr("class", "line")
      .attr("d", function(d) {return line(d.data); })
      .style("stroke-width", function(d) {return d.stroke_width == undefined ? 4: d.stroke_width;})
      .style("fill", "none")
      .style("stroke", "url(#line-gradient)");
      
    this.gradientLines.append("title")
		.text(function(d) { return d.name;});
		
	var legend = this.svg.selectAll("legend")
		.data(color.domain().slice().reverse())
		.enter().append("g")
		.attr("class", "legend")
		.attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });
		
	legend.append("rect")
		.attr("x", margin.left + 6)
		.attr("y", height - 1.5*margin.bottom)
		.attr("width", 18)
		.attr("height", 18)
		.style("fill", color);

	legend.append("text")
		.attr("x", margin.left + 4 )
		.attr("y", height - 1.5*margin.bottom + 7)
		.attr("dy", ".35em")
		.style("text-anchor", "end")
		.text(function(d) { return d; });
}

BaseChart.prototype.draw_circles = function() {
	var xScale = this.xScale;
	var yScale = this.yScale;
	this.circles.append("circle")
      .attr("class", "dot")
      .attr("r", function(d) { return d.r;})
      .attr("cx", function(d) { return xScale(d.x); })
      .attr("cy", function(d) { return yScale(d.y); })
      .style("fill", function(d) { return get_random_color(d.name); });

    this.circles.append("title")
		.text(function(d) { return d.title;});
}

BaseChart.prototype.draw_dimensions = function() {
	var line = d3.svg.line();
	var xScale = this.xScale;
	var yScale = this.yScale;
	var dims = this.dims;
	this.dimensions.append("g")
            .attr("class", "line")
            .append("path")
	        .attr("d", function (d) {
						  return line(dims.map(function(p) {return [xScale(p), yScale[p](d[p])]; }));
					  })
            .style("stroke",function(d) {return get_random_color(d.name); })
            .style("stroke-width", function(d) {return d.stroke_width == undefined ? 3 : d.stroke_width; })
            .style("stroke-dasharray", function(d) {if (d.name == "avg") return ("3, 3"); });
	
    this.dimensions.append("title").text(function(d) { return d.name;});  
}

/////////////////// Line chart /////////////////////
function LineChart(options) {
    
	var o = {};
	if (options.logScale != undefined) {
		options.shapes.lines.forEach(function(l) {
			l.data.forEach(function(d) {if (d.y == 0) { d.y = NaN;}}); // log(0) = -Infty --> don't display
		});
	}
	options.shapes.lines.forEach(function(l) {
		getMinsMaxs(l.data, ['x','y'], o);
	});
	options.xScale = getTimeScale(o.min_x, o.max_x);
	if (options.logScale != undefined) {
		options.yScale = getLogScale([height, 0], [o.min_y, o.max_y]);
	} else {
		options.yScale = getLinearScale([height, 0], [o.min_y, o.max_y]);
	}	
        BaseChart.call(this, options);
}
LineChart.prototype = Object.create(BaseChart.prototype);

/////////////////// Line chart /////////////////////
function LineChart2(options) {
	var o = {};
	
	if (options.logScale != undefined) {
		options.shapes.lines.forEach(function(l) {
			l.data.forEach(function(d) {if (d.y == 0) { d.y = NaN;}}); // log(0) = -Infty --> don't display
		});
	}
	options.shapes.lines.forEach(function(l) {
		getMinsMaxs(l.data, ['x','y'], o);
	});
	options.xScale = getLinearScale([0, width], [o.min_x, o.max_x]);
	if (options.logScale != undefined) {
		options.yScale = getLogScale([height, 0], [o.min_y, o.max_y]);
	} else {
		options.yScale = getLinearScale([height, 0], [0, o.max_y]);
	}	
	BaseChart.call(this, options);
}
LineChart2.prototype = Object.create(BaseChart.prototype);





/////////////////// Gantt chart /////////////////////
function GanttChart(options) {
    console.log("GC:", options);
	var o = {};
	getMinsMaxs(options.shapes.rectangles, ['x', 'y'], o);
	options.shapes.rectangles.sort(function(a, b) { return b.end - a.end; });
	//options.xScale = getTimeScale(o.min_x, o.max_x);
        options.xScale = getLinearScale([0, width], [0, 84600]);
        options.yScale = getLinearScale([height, 0], [0, 1]);
    BaseChart.call(this, options);
}
GanttChart.prototype = Object.create(BaseChart.prototype);
  
GanttChart.prototype.appendGridding = function() { 
	// empty
}
  
  
/////////////////// Area + Line chart /////////////////////
function AreaLineChart(options) {
	var o = {};
	options.shapes.areas.forEach(function(d) {
		getMinsMaxs(d.data, ['x', 'y0', 'y1'], o);
	});
	options.shapes.lines[0].stroke_width = 2;
	options.xScale = getTimeScale(o.min_x, o.max_x);
	options.yScale = getLinearScale([height, 0], [o.min_y0, o.max_y1]);
	
	BaseChart.call(this, options);
}
AreaLineChart.prototype = Object.create(BaseChart.prototype);


/////////////////// StackedArea chart /////////////////////
function StackedAreaChart(options) {
	var stack = d3.layout.stack().values(function(d) { return d.data; });
	stack(options.shapes.stackedAreas);
	
	var o = {};
	options.shapes.stackedAreas.forEach(function(d) {
		getMinsMaxs(d.data, ['x', 'y0', 'y'], o);
	});
	
	options.shapes.stackedAreas.forEach(function(d) {
		d.data.forEach(function(s) {
			if (o.max_y0 < s.y+s.y0){
				o.max_y0 = s.y+s.y0;
			}
		});
	});
	options.xScale = getTimeScale(o.min_x, o.max_x);
	options.yScale = getLinearScale([height, 0], [o.min_y0, o.max_y0]);
	BaseChart.call(this, options);
}
StackedAreaChart.prototype = Object.create(BaseChart.prototype);

/////////////////// Bar chart /////////////////////
function BarChart(options) {
	var o = {};
	if (options.sort_bars === undefined) {
		options.shapes.bars.sort(function(a, b) { return b.y - a.y; });
	}
	getMinsMaxs(options.shapes.bars, ['y'], o);
	options.xScale = getOrdinalScale(options.shapes.bars.length);
	options.xScale.domain(options.shapes.bars.map(function(b){return b.name;}));
	options.yScale = getLinearScale([height, 0], [0, o.max_y]);
	options.rotate_x_ticks = true;
        BaseChart.call(this, options);
}
BarChart.prototype = Object.create(BaseChart.prototype);

/////////////////// GradientLine chart /////////////////////
function GradientLineChart(options) {
	var o = {};
	options.shapes.gradientLines.forEach(function(l) {
		getMinsMaxs(l.data, ['x','y'], o);
	});
	options.xScale = getTimeScale(o.min_x, o.max_x);
	options.yScale = getLinearScale([height, 0], [o.min_y, o.max_y]);
	options.shapes.gradientLines[0].stroke_width = 4;
	BaseChart.call(this, options);
}
GradientLineChart.prototype = Object.create(BaseChart.prototype);

GradientLineChart.prototype.appendGridding = function() {
	this.yScale.domain([0, Math.max(1, this.yScale.domain()[1])]);
	LineChart.prototype.appendGridding.call(this);
}

////////////////// Scatter chart /////////////////////
function ScatterChart(options) {
	var o = {};
	getMinsMaxs(options.shapes.circles, ['x','y'], o);
	options.xScale = getLinearScale([0, width], [0.9*o.min_x, 1.1*o.max_x]);
        if (options.logScale != undefined) {
		options.yScale = getLogScale([height, 0], [o.min_y, o.max_y]);
	} else {
		options.yScale = getLinearScale([height, 0], [0, o.max_y]);
	}
	//options.yScale = getLinearScale([height, 0], [0.9*o.min_y, 1.1*o.max_y]);
	BaseChart.call(this, options);
}
ScatterChart.prototype = Object.create(BaseChart.prototype);

////////////////// Scatter chart /////////////////////
function ScatterChart2(options) {
	var o = {};
	getMinsMaxs(options.shapes.circles, ['x','y'], o);
        
        if (options.shapes.lines) {
            options.shapes.lines.forEach(function(l) {
		getMinsMaxs(l.data, ['x','y'], o);
            });
        }
        
        options.xScale = getTimeScale(o.min_x, o.max_x);
	options.yScale = getLinearScale([height, 0], [0.9*o.min_y, 1.1*o.max_y]);
	BaseChart.call(this, options);
}
ScatterChart2.prototype = Object.create(BaseChart.prototype);

////////////////// Parallel chart /////////////////////
function ParallelChart(options) {
	options.xScale = d3.scale.ordinal().rangePoints([0, width], 1);
	options.yScale = {};
	var dims;
	options.shapes.dimensions.forEach(function(d) {
        // Extract the list of dimensions and create a scale for each.
        options.xScale.domain(dims = d3.keys(d).filter(function(k) {
            return k != "name" && (options.yScale[k] = getLinearScale([height, 0], [0, 1.1* d3.max(options.shapes.dimensions, function(p) { return +p[k]; })])); 
        }));
	});	
	options.dims = dims;
	//BaseChart.call(this, title, div, {dimensions:data}, xScale, yScale, undefined, undefined, undefined, undefined, url, undefined, undefined);
	BaseChart.call(this, options);
}
ParallelChart.prototype = Object.create(BaseChart.prototype);

ParallelChart.prototype.appendGridding = function() { 
	// empty
}

ParallelChart.prototype.appendXAxis = function() {
// empty
}

ParallelChart.prototype.appendYAxis = function() {
	var xScale = this.xScale;
	var yScale = this.yScale;
	// Add a group element for each dimension.
    var g = this.svg.selectAll(".dimension")
            .data(this.dims)
            .enter().append("g")
            .attr("class", "dimension")
            .attr("transform", function(d) { return "translate(" + xScale(d) + ")"; });
	// Add an axis and title.
	g.append("g")
		.attr("class", "axis")
        .each(function(d) { d3.select(this).call(d3.svg.axis().orient("left").scale(yScale[d])); })
        .append("text")
        .attr("text-anchor", "middle")
        .attr("y", height + (margin.bottom/4))
        .text(String);
}


/////////////////// BoxPlot chart /////////////////////
function BoxPlotChart(options) {
	var o = {};
	options.shapes.lines.forEach(function(l) {
		getMinsMaxs(l.data, ['x','y'], o);
	});
	options.xScale = getTimeScale(o.min_x, o.max_x);
	options.yScale = getLinearScale([height, 0], [o.min_y, o.max_y]);
	BaseChart.call(this, options);
}
BoxPlotChart.prototype = Object.create(BaseChart.prototype);

var chart = {
	random_color: function(name) { return get_random_color(name);},
	line: function(opts) { return new LineChart(opts);},
	bar: function(opts) { return new BarChart(opts);},
	parallel: function(opts) { return new ParallelChart(opts);},
	gantt: function(opts) {	return new GanttChart(opts);},
	arealine: function(opts) {return new AreaLineChart(opts);},
	gradientline: function(opts) {return new GradientLineChart(opts);},
	stackedarea: function(opts) {return new StackedAreaChart(opts);},
	scatter: function(opts) {return new ScatterChart(opts);}
}

