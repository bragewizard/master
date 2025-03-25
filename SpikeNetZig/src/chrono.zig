
//! Chrono.zig - Simple timing functionality for measuring execution time
//! Translated from the original C++ Chrono implementation

const std = @import("std");
const Allocator = std.mem.Allocator;
const time = std.time;

/// Chrono provides timing functionality for measuring code execution
pub const Chrono = struct {
    allocator: *Allocator,
    start_time: i64,
    stop_time: i64,
    is_running: bool,
    
    /// Initialize a new Chrono instance
    pub fn init(allocator: *Allocator) !*Chrono {
        var self = try allocator.create(Chrono);
        self.allocator = allocator;
        self.start_time = 0;
        self.stop_time = 0;
        self.is_running = false;
        return self;
    }
    
    /// Start the timer
    pub fn start(self: *Chrono) void {
        self.start_time = time.milliTimestamp();
        self.is_running = true;
    }
    
    /// Stop the timer
    pub fn stop(self: *Chrono) void {
        self.stop_time = time.milliTimestamp();
        self.is_running = false;
    }
    
    /// Read the elapsed time in milliseconds
    pub fn read(self: *Chrono) i64 {
        if (self.is_running) {
            return time.milliTimestamp() - self.start_time;
        } else {
            return self.stop_time - self.start_time;
        }
    }
    
    /// Free resources
    pub fn deinit(self: *Chrono) void {
        self.allocator.destroy(self);
    }
    
    /// Legacy compatibility with C++ Start method
    pub fn Start(self: *Chrono) void {
        self.start();
    }
    
    /// Legacy compatibility with C++ Stop method
    pub fn Stop(self: *Chrono) void {
        self.stop();
    }
    
    /// Legacy compatibility with C++ Read method
    pub fn Read(self: *Chrono) i64 {
        return self.read();
    }
};
